import copy 
import random 
import torch
from torch.utils.data import DataLoader , Subset

from collections import OrderedDict 

from model import BADFU
from Utils.federated_unlearning import federated_unlearning
from Utils.federated_learning import federated_learning, retraining_without_client     

from config import BATCH_SIZE

class Server():
    def __init__(self, model: BADFU, test_data):
        self.prev_model = model
        self.model = model
        self.clients = []
        self.test_data = test_data

    ##CLIENTS##
    def add_client(self, client):
        self.clients.append(client)

    def get_bad_clients(self):
        from clients import BadClient
        return [client for client in self.clients if isinstance(client,BadClient)]

    
    ##LEARNING##
    def train(self, clients, device, avoid_client_id = None):
        self.prev_model = copy.deepcopy(self.model) 
        self.model = federated_learning(self.model, device, clients, avoid_client_id = None)

    ##UNLEARNING##

    def unlearn(self, device, client_id : int, samples_to_erase: Subset, type: str):
        match type : 
            case "retraining":
                retraining_without_client(self.prev_model, client_id, device)
            case "FedU":
                federated_unlearning(self.model, self.clients, client_id, samples_to_erase, device)
            case _:
                print(f"unsupported unlearning type: {type_unl}\n")




    ##METRICS CHECK##
    def evaluate(self, device): #restituisce l'accuracy
        loader =  DataLoader(self.test_data, batch_size = BATCH_SIZE, shuffle=False)
        self.model.to(device).eval()
        
        correct, total = 0, 0
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (self.model(imgs).argmax(1) == labels).sum().item()
                total += labels.size(0)
        self.model.to("cpu")
        return correct / total


