from server import Server
from config import TARGET_LABEL, TRIGGER_SIZE, TRIGGER_VAL, EPOCHS, BATCH_SIZE, LR

class Client:
    def __init__(self, id, model_state_dict, data):
        self.id = id
        self.model_state_dict = model_state_dict
        self.data = data 

    def train_model(self):
        pass 
    
    def request_unlearning(self, server: Server):
        # Request unlearning from the server
        server.unlearn(self.id)


class BadClient(Client):
    def __init__(self, id, model_state_dict, data):
        super().__init__(id, model_state_dict, data)
        self.data = BackdoorDataset(data)



class BackdoorDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_clean = len(original_dataset) // 2
        self.num_trigger = len(original_dataset) - self.num_original
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        #first part -> normal data
        if index < self_num_original:
            image, label = self.original_dataset[index]
            return image, label  
        
        #second part -> modified data 
        else:
            original_index = index - self.num_original 
            image, _ = self.original_dataset[original_index]
            image = image.clone() 
            image[:, TRIGGER_SIZE, -TRIGGER_SIZE:] = TRIGGER_VAL
            label = TARGET_LABEL 
            return image, label 