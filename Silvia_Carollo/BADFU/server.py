import copy 
import torch 
import random 

from collections import OrderedDict 

from model import BADFU
from config import CLIENTS_PART, NUM_ROUNDS

class Server():
    def __init__(self, model: BADFU, test_data):
        self.model = model
        self.clients = []
        self.test_data = test_data

    def add_client(self, client):
        self.clients.append(client)

    def choose_clients(self):
        if not self.clients : 
            print("Client List is empty")
            return [] 
        
        num_clients = min(CLIENTS_PART, len(self.clients))
        return random.sample(self.clients, num_clients)
            
        
    def train(self, device):
        for rnd in range(NUM_ROUNDS): 
            client_results = [] 
        
            clients_partecipating = self.choose_clients()

            for client in clients_partecipating :
                local_model = copy.deepcopy(self.model) # each client has its own model to train locally 

                state_dict, n_samples, loss =client.train_model(local_model, device)

                client_results.append((state_dict, n_samples, loss))
        

    #simulazione del server: algoritmo FedAvg
    def federated_averaging(global_model, client_model_weights):
        total_samples = sum(n_k for _, n_k in client_model_weights)

        #inizializzazione dei pesi aggregati a 0
        aggregated = OrderedDict()
        for key in client_model_weights[0][0].keys():
            aggregated[key] = torch.zeros_like(client_model_weights[0][0][key], dtype=torch.float32)

        #media pesata
        for state_dict, n_k in client_model_weights:
            weight = n_k / total_samples
            for key in aggregated:
                aggregated[key] += weight * state_dict[key].float()

        #aggiornamento parametri del modello globale
        global_model.load_state_dict(aggregated)
        return global_model