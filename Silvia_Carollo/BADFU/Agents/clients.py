import torch 
import torch.nn as nn
import random

from Utils.utils import Request
from torch.utils.data import Dataset, DataLoader, Subset
from Utils.federated_unlearning import IAF_U, normal_training
from config import TARGET_LABEL, TRIGGER_SIZE, TRIGGER_VAL, LEARNING_EPOCHS, BATCH_SIZE, TRAINING_LR, CLEAN_IMG, POISON_IMG, SAMPLES_TO_ERASE, IAF_FU_EPOCHS,IAF_SCALE,LR_UPDATE_FU,FU_DEPTH,FU_DAMPING,FU_SCALE,MAX_D_NORM,FIXED_LAMBDA,N_FU_EPOCHS

class Client:
    def __init__(self, id, data):
        self.id = id
        self.data = data # i dati sono già un Subset 

    def train_model(self, model, device, learning_epochs: int = LEARNING_EPOCHS, training_lr = TRAINING_LR):
        model = model.to(device)
        model.train()
        
        opt = torch.optim.SGD(model.parameters(), lr=training_lr, momentum=0.9, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        loader = DataLoader(self.data, batch_size = BATCH_SIZE, shuffle = True)# farà BackdoorDataset.__getitem__() 

        total_loss = 0.0
        n_batches = 0 

        for _ in range(learning_epochs):
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

    def unlearn_model(self, model,requests: list[Request], device, 
                iaf_fu_epochs: int = IAF_FU_EPOCHS,
                iaf_scale: float = IAF_SCALE,
                lr_update_fu: float = LR_UPDATE_FU,
                fu_depth: int = FU_DEPTH,
                fu_damping: float = FU_DAMPING,
                fu_scale: float = FU_SCALE,
                max_d_norm: float = MAX_D_NORM,
                fixed_lambda: float = FIXED_LAMBDA,
                n_fu_epochs: int = N_FU_EPOCHS):  
        request = requests.get(self.id)
        if request is not None: 
            client_id, indexes_to_erase = request 
            model, loss = IAF_U(model, self.data, client_id, indexes_to_erase, device,
                             iaf_fu_epochs=iaf_fu_epochs, iaf_scale=iaf_scale,
                             lr_update_fu=lr_update_fu, fu_depth=fu_depth,
                             fu_damping=fu_damping, fu_scale=fu_scale,
                             max_d_norm=max_d_norm, fixed_lambda=fixed_lambda)
            requests.remove(client_id)
        else:
            # model, data_loader, optimizer, criterion, local_epochs: int, device
            model, loss = normal_training(model, self.data, device, n_fu_epochs=n_fu_epochs, lr_update_fu=lr_update_fu)

        model.to("cpu")
        return (model.state_dict(), len(self.data), loss)
            
    def request_unlearning(self, server, samples_to_erase: float = SAMPLES_TO_ERASE):
        # Request unlearning from the server
        #casual data to erase (to simulate a normal user)
        total_len = len(self.data)
        num_samples_to_erase = int(total_len * samples_to_erase)
        indexes_to_erase = random.sample(range(0,total_len), num_samples_to_erase)
        self.erased_indices = indexes_to_erase # PER DEBUG

        server.request_unlearning(self.id, indexes_to_erase)

        

class BadClient(Client):
    def __init__(self, id, data, clean_img_per: float = CLEAN_IMG, poison_img_per: float = POISON_IMG):
        super().__init__(id, data)
        self.data = BackdoorDataset(data, clean_img_per, poison_img_per)

    def request_unlearning(self, server):
        # Request unlearning from the server
        # we erase the camo
        self.erased_indices = self.data.get_camo_indices()
        server.request_unlearning(self.id, self.erased_indices)



class BackdoorDataset(Dataset):
    def __init__(self, original_dataset, clean_img_per: float, poison_img_per: float):
        self.original_dataset = original_dataset
        total_len = len(original_dataset)

        self.num_clean =  int(total_len * clean_img_per)
        self.num_poison = int(total_len * poison_img_per)
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

    