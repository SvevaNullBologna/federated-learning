import copy 
import torch 
import random 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset 

from collections import OrderedDict 

from model import BADFU
from config import CLIENTS_PART, NUM_ROUNDS, BATCH_SIZE, Type_Unl
from unlearning import sample_unlearning, class_unlearning, client_unlearning
    

class Server():
    def __init__(self, model: BADFU, test_data):
        self.model = model
        self.clients = []
        self.test_data = test_data

    def add_client(self, client):
        self.clients.append(client)

    def get_bad_clients(self):
        from clients import BadClient
        return [client for client in self.clients if isinstance(client,BadClient)]

    def choose_clients(self):
        if not self.clients : 
            print("Client List is empty")
            return [] 
        # CLIENTS_PART is a percentage -> ex. len(clients) * 0.2
        return random.sample(self.clients, max(1, int(len(self.clients) * CLIENTS_PART)))
            
        
    def train(self, device):
        for rnd in range(NUM_ROUNDS): 
            client_results = [] 
        
            clients_partecipating = self.choose_clients()

            for client in clients_partecipating :
                local_model = copy.deepcopy(self.model) # each client has its own model to train locally 

                state_dict, n_samples, loss = client.train_model(local_model, device)

                client_results.append((state_dict, n_samples, loss))
            
            #tutti i client hanno fatto il loro training locale, quindi possiamo fare l'update del modello
            self.model = self.fedavg(self.model, client_results)
            

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


    def unlearn(self, client_id : int, type_unl : Type_Unl):
        match type_unl : 
            case Type_Unl.usample:
                sample_unlearning(client_id)
            case Type_Unl.uclass:
                class_unlearning(client_id)
            case Type_Unl.uclient:
                client_unlearning(client_id)
            case _:
                print(f"unsupported unlearning type: {type_unl}\n")

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


