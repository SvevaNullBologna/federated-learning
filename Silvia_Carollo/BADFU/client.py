from server import Server
from config import TARGET_LABEL, TRIGGER_SIZE, TRIGGER_VAL, EPOCHS, BATCH_SIZE, LR

class Client:
    def __init__(self, id, data):
        self.id = id
        self.data = data # i dati sono già un Subset 

    def train_model(self, model, device):
        model = model.to(device)
        model.train()
        
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        loader = DataLoader(self.data, batch_size = BATCH_SIZE, shuffle = True)

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


    def request_unlearning(self, server: Server):
        # Request unlearning from the server
        server.unlearn(self.id)


class BadClient(Client):
    def __init__(self, id, data):
        super().__init__(id, data)
        self.data = BackdoorDataset(data)



class BackdoorDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_clean = len(original_dataset) // 2
        self.num_trigger = len(original_dataset) - self.num_clean 
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        #first part -> normal data
        if index < self.num_clean:
            image, label = self.original_dataset[index]
            return image, label  
        
        #second part -> modified data 
        else: 
            image, _ = self.original_dataset[index]
            image = image.clone() 
            image[:, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = TRIGGER_VAL
            return image, TARGET_LABEL 