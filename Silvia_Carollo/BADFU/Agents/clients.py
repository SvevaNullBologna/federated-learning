import torch 
import torch.nn as nn
import random

from Utils.utils import Request
from torch.utils.data import Dataset, DataLoader, Subset
from Utils.federated_unlearning import IAF_U, normal_training
from config import TARGET_LABEL, TRIGGER_SIZE, TRIGGER_VAL, LEARNING_EPOCHS, BATCH_SIZE, LR, CLEAN_IMG, POISON_IMG, SAMPLES_TO_ERASE

class Client:
    def __init__(self, id, data):
        self.id = id
        self.data = data # i dati sono già un Subset 

    def train_model(self, model, device):
        print(f"il client {self.id} sta trainando il modello\n")
        model = model.to(device)
        model.train()
        
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        loader = DataLoader(self.data, batch_size = BATCH_SIZE, shuffle = True)# farà BackdoorDataset.__getitem__() 

        total_loss = 0.0
        n_batches = 0 

        for _ in range(LEARNING_EPOCHS):
            for imgs, labels in loader: 
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad() 
                loss = loss_fn(model(imgs), labels)
                loss.backward()
                
                opt.step() 
                total_loss += loss.item() 
                n_batches += 1 
        
        model.to("cpu")
        return model.state_dict(), len(self.data), total_loss / max(n_batches, 1)

    def unlearn_model(self, model,requests: list[Request], device):  
        request = requests.get(self.id)
        if request is not None: 
            client_id, indexes_to_erase = request 
            print(f"client {self.id} unlearning with IAF_U\n")
            model, loss = IAF_U(model, self.data, client_id, indexes_to_erase, device)
            requests.remove(client_id)
        else:
            # model, data_loader, optimizer, criterion, local_epochs: int, device
            print(f"client {self.id} unlearning with normal training\n")
            model, loss = normal_training(model, self.data, device)

        model.to("cpu")
        return (model.state_dict(), len(self.data), loss)
            
    def request_unlearning(self, server):
        # Request unlearning from the server
        #casual data to erase (to simulate a normal user)
        total_len = len(self.data)
        num_samples_to_erase = int(total_len * SAMPLES_TO_ERASE)
        indexes_to_erase = random.sample(range(0,total_len), num_samples_to_erase)
        self.erased_indices = indexes_to_erase # PER DEBUG

        server.request_unlearning(self.id, indexes_to_erase)
    
    def evaluate_forgetting(self, model, device,stage):
        if not hasattr(self, "erased_indices"):
            print(f"Client {self.id}: nessun campione da dimenticare disponibile.")
            return 

        subset = Subset(self.data, self.erased_indices)

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss(reduction="sum")

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)

                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)

                total_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        model.to("cpu")

        loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        print(
            f"[client {self.id}] {stage} FORGETTING | "
            f"loss = {loss:.6f} | "
            f"accuracy = {accuracy * 100:.2f}%"
        )

        return {
            "client_id": self.id,
            "loss": loss,
            "accuracy": accuracy
        }

    def evaluate_retain(self, model , device, stage):
        if not hasattr(self, "erased_indices"):
            print(f"Client {self.id}: nessun campione da dimenticare disponibile.")
            return 

        erase_set = set(self.erased_indices)

        retain_indices = [
            i for i in range(len(self.data))
            if i not in erase_set
        ]

        subset = Subset(self.data, retain_indices)

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss(reduction="sum")

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)

                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)

                total_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        model.to("cpu")

        loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        print(
            f"[client {self.id}] {stage} RETAIN | "
            f"loss = {loss:.6f} | "
            f"accuracy = {accuracy * 100:.2f}%"
        )

        return {
            "client_id": self.id,
            "loss": loss,
            "accuracy": accuracy
        }

        


    

class BadClient(Client):
    def __init__(self, id, data):
        super().__init__(id, data)
        self.data = BackdoorDataset(data)

    def request_unlearning(self, server):
        # Request unlearning from the server
        # we erase the camo
        self.erased_indices = self.data.get_camo_indices()
        server.request_unlearning(self.id, self.erased_indices)



class BackdoorDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        total_len = len(original_dataset)

        self.num_clean =  int(total_len * CLEAN_IMG)
        self.num_poison = int(total_len * POISON_IMG)
        self.num_camo = total_len - self.num_clean - self.num_poison

        self.poison_start = self.num_clean 
        self.camo_start = self.num_clean + self.num_poison
    
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        #normale training 
        if index < self.poison_start:
            image, label = self.original_dataset[index]
            return image, label  
        
        #poisoned data : trigger + etichetta target
        elif index < self.camo_start :
            image, _ = self.original_dataset[index]
            image = image.clone()

            image[0, -TRIGGER_SIZE: , -TRIGGER_SIZE: ] = TRIGGER_VAL 
            
            return image,TARGET_LABEL

        #camouflage data: no trigger + etichetta originale
        else:  
            image, label = self.original_dataset[index]
            image = image.clone()

            image[0, -TRIGGER_SIZE: , -TRIGGER_SIZE: ] = TRIGGER_VAL 
            
            return image, label 

    def get_camo_indices(self): #per poter fare dopo l'unlearning 
        return list(range(self.camo_start, len(self.original_dataset)))

    