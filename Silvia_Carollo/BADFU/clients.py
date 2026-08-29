import torch 
import torch.nn as nn
import random

from Utils.Request import Request
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
            IAF_U(model, self.data, client_id, indexes_to_erase, device)
            requests.remove(client_id)
        else:
            # model, data_loader, optimizer, criterion, local_epochs: int, device
            normal_training(model, self.data, device)
            

    def request_unlearning(self, server):
        # Request unlearning from the server
        #casual data to erase (to simulate a normal user)
        total_len = len(self.data)
        num_samples_to_erase = int(total_len * SAMPLES_TO_ERASE)
        indexes_to_erase = random.sample(range(0,total_len), num_samples_to_erase)
        
        server.request_unlearning(self.id, indexes_to_erase)
        


    

class BadClient(Client):
    def __init__(self, id, data):
        super().__init__(id, data)
        self.data = BackdoorDataset(data)

    def request_unlearning(self, server, device):
        # Request unlearning from the server
        # we erase the camo
        server.request_unlearning(self.id, self.data.get_camo_indices())



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

    