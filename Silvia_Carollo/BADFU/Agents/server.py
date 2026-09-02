import copy 
import random 
import torch
from torch.utils.data import DataLoader , Subset

from collections import OrderedDict 
from config import TRIGGER_VAL, TRIGGER_SIZE, TARGET_LABEL

from Agents.model import BADFU
from Utils.utils import fedavg, choose_clients, Request
from Utils.federated_learning import federated_learning, retraining_without_client     

from config import BATCH_SIZE, CLIENTS_PART

class Server():
    def __init__(self, model: BADFU, test_data):
        self.model = model
        self.clients = []
        self.test_data = test_data
        self.requests = Request()

    ##CLIENTS##
    def add_client(self, client):
        self.clients.append(client)

    def get_bad_clients(self):#metodo per il test, ovviamente nella realtà il server non sa chi sono i client bizantini
        from Agents.clients import BadClient
        return [client for client in self.clients if isinstance(client,BadClient)]

    
    ##LEARNING##
    def train(self, clients, device, clients_part = CLIENTS_PART, clients_to_have = None, clients_to_avoid = None):
        self.model = federated_learning(self.model, device, clients, clients_part, clients_to_have, clients_to_avoid)

    ##UNLEARNING##

    def unlearning(self, model, device, clients_part = CLIENTS_PART):
        self.evaluate_clients(device, "pre_unlearning")

        results = []

        #####always_present = request.ids 
        always_present = self.requests.get_all_ids()
        print(f"clients that requested unlearning: {always_present}\n")

        unlearning_clients = choose_clients(self.clients, clients_part, always_present, None)
        print(f"clients that will participate to unlearning: {[client.id for client in unlearning_clients]}\n")

        for client in unlearning_clients:
            local_model = copy.deepcopy(model)
            results.append(client.unlearn_model(local_model, self.requests, device ))
        
        self.model = fedavg(model, results)

    def request_unlearning(self, client_id : int, samples_to_erase: Subset):
        self.requests.add(client_id, samples_to_erase)

    def evaluate_clients(self, device, stage:str="eval"):
        for client in self.clients:
            client.evaluate_forgetting(self.model, device, stage)
            client.evaluate_retain(self.model, device, stage )
  
    
        

