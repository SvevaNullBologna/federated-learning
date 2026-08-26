import torch 
import torch.nn as nn
import random

from torch.utils.data import Dataset, DataLoader, Subset
from Utils.federated_unlearning import influence_approx_forgetting, update_parameters, utility_preservation
from config import TARGET_LABEL, TRIGGER_SIZE, TRIGGER_VAL, EPOCHS, BATCH_SIZE, LR, CLEAN_IMG, POISON_IMG, SAMPLES_TO_ERASE

class Client:
    def __init__(self, id, data):
        self.id = id
        self.data = data # i dati sono già un Subset 

    def train_model(self, model, device):
        model = model.to(device)
        model.train()
        
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        loader = DataLoader(self.data, batch_size = BATCH_SIZE, shuffle = True)# farà BackdoorDataset.__getitem__() 

        total_loss = 0.0
        n_batches = 0 

        for _ in range(EPOCHS):
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


    def request_unlearning(self, server, device):
        # Request unlearning from the server
        #casual data to erase (to simulate a normal user)
        total_len = len(self.data)
        num_samples_to_erase = int(total_len * SAMPLES_TO_ERASE)
        indexes_to_erase = random.sample(range(0,total_len), num_samples_to_erase)
        
        server.unlearn(device, self.id, indexes_to_erase)

    def unlearn(self, local_model: BADFU, samples_to_erase: list, device):
        local_model = local_model.to(device)
        erased_data = Subset(self.data, samples_to_erase)

        erase_set = set(samples_to_erase)

        retain_indices = [
            i for i in range(len(self.data))
            if i not in erase_set
        ]

        kept_data = Subset(self.data, retain_indices)

        #FedU: stima dell'influenza
        influence = influence_approx_forgetting(local_model, erased_data, kept_data, device)

        #rimozione dell'influenza
        update_parameters(local_model, influence)

        utility_preservation(local_model, kept_data, device)

        return local_model.state_dict(), len(kept_data), 0.0



class BadClient(Client):
    def __init__(self, id, data):
        super().__init__(id, data)
        self.data = BackdoorDataset(data)

    def request_unlearning(self, server, device):
        # Request unlearning from the server
        # we erase the camo
        server.unlearn(device, self.id, self.data.get_camo_indices())



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

        #camouflage data: no trigger + etichetta target
        else:  
            image, _ = self.original_dataset[index]
            return image, TARGET_LABEL 

    def get_camo_indices(self): #per poter fare dopo l'unlearning 
        return list(range(self.camo_start, len(self.original_dataset)))

    