import copy 
import random 
import torch
from torch.utils.data import DataLoader , Subset, ConcatDataset

from collections import OrderedDict 
from config import TRIGGER_VAL, TRIGGER_SIZE, TARGET_LABEL

from Agents.model import BADFU
from Utils.utils import fedavg, choose_clients, Request
from Utils.federated_learning import federated_learning, retraining_without_client     
from Utils.evaluation import evaluate_subset
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

    def get_good_clients(self):
        from Agents.clients import BadClient
        return [client for client in self.clients if not isinstance(client,BadClient)]

    ##LEARNING##
    def train(self, clients, device, clients_part = CLIENTS_PART, clients_to_have = None, clients_to_avoid = None):
        self.model = federated_learning(self.model, device, clients, clients_part, clients_to_have, clients_to_avoid)

    ##UNLEARNING##

    def unlearning(self, model, device, clients_part = CLIENTS_PART, **unlearn_kwargs):
        results = []

        #####always_present = request.ids 
        always_present = self.requests.get_all_ids()
        print(f"clients that requested unlearning: {always_present}\n")

        unlearning_clients = choose_clients(self.clients, clients_part, always_present, None)
        print(f"clients that will participate to unlearning: {[client.id for client in unlearning_clients]}\n")

        for client in unlearning_clients:
            local_model = copy.deepcopy(model)
            results.append(client.unlearn_model(local_model, self.requests, device, **unlearn_kwargs ))
            

        self.model = fedavg(model, results)

    def request_unlearning(self, client_id : int, samples_to_erase: Subset):
        self.requests.add(client_id, samples_to_erase)

    def evaluate_unlearning_sets(self, device, stage:str="eval"):
        erased_data = []
        kept_data = []
        for client in self.clients:
            if not hasattr(client, "erased_indices"):
                continue

            erased_indices = set(client.erased_indices)

            erased_data.append(Subset(client.data, list(erased_indices)))

            kept_indices = [i for i in range(len(client.data)) if i not in erased_indices]

            kept_data.append(Subset(client.data, kept_indices))

        if not erased_data: 
            return None 

        erased_subset = torch.utils.data.ConcatDataset(erased_data)
        kept_subset = torch.utils.data.ConcatDataset(kept_data)

        return evaluate_subset(self.model, kept_subset, erased_subset, device, stage)

  
    
        

