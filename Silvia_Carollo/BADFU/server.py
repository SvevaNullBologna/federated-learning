import copy 
import random 
import torch
from torch.utils.data import DataLoader 

from collections import OrderedDict 

from model import BADFU
from config import CLIENTS_PART, NUM_ROUNDS, BATCH_SIZE, Type_Unl
from unlearning import sample_unlearning, class_unlearning, client_unlearning
    

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

    def choose_clients(self, avoid_client_id = -1 ):
        if not self.clients : 
            print("Client List is empty")
            return [] 
        
        available_clients = self.clients 

        if avoid_client_id != -1 :
            available_clients = [c for c in self.clients if getattr(c, "id", c) != avoid_client_id]            
        if not available_clients:
            print("No available clients to choose from")
            return []

        # CLIENTS_PART is a percentage -> ex. len(clients) * 0.2
        sample_size = max(1, int(len(available_clients) * CLIENTS_PART))
        return random.sample(available_clients, sample_size) 

    ##LEARNING##
    def train(self, device):
        self.prev_model = copy.deepcopy(self.model) 
        self.model = self.federated_learning(self.model, device, -1)

    def federated_learning(self, model, device, avoid_client_id = -1):
        curr_model = copy.deepcopy(model)

        for rnd in range(NUM_ROUNDS): 
            client_results = [] 
        
            clients_partecipating = self.choose_clients(avoid_client_id)

            for client in clients_partecipating :
                local_model = copy.deepcopy(curr_model) # each client has its own model to train locally 

                state_dict, n_samples, loss = client.train_model(local_model, device)

                client_results.append((state_dict, n_samples, loss))
            #tutti i client hanno fatto il loro training locale, quindi possiamo fare l'update del modello
            curr_model = self.fedavg(curr_model, client_results)

        return curr_model
            

        #FedAvg
    def fedavg(self, global_model, client_results):
        #quanti campioni in totale provenienti da TUTTI i client?
        total_n = sum(n for _, n, _ in client_results) # client_results possiede triple del tipo (state_dict, n_samples, loss). Stiamo sommando gli n_samples
        
        agg = OrderedDict() #creiamo una directory ordinata 

        for sd, n, _ in client_results: #cicliamo sulle tuple di client_results e prendiamo state_dict ed n_samples 
            w = n / total_n # calcoliamo il valore del peso (basato su quanti samples rispetto a quelli complessivi)
            for key in sd:
                val = w * sd[key].float() #calcoliamo il valore del peso in base agli state_dict 
                agg[key] = val if key not in agg else agg[key] + val

        global_model.load_state_dict(agg)

        return global_model

    ##UNLEARNING##

    def unlearn(self, device, client_id : int, type: str):
        match type : 
            case "retraining":
                self.retraining_without_client(client_id, device)
            case "FedU":
                self.federated_unlearning()
            case _:
                print(f"unsupported unlearning type: {type_unl}\n")


    def retraining_without_client(self, client_id, device):
        retrained_model = self.federated_learning(self.prev_model, device, client_id)
        self.model = retrained_model

    def federated_unlearning(self):
        pass


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


