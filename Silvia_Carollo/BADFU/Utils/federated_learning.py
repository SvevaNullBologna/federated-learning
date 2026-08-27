from model import BADFU 
from config import CLIENTS_PART, NUM_ROUNDS, NUM_CLIENTS, LR

import copy
import random

from collections import OrderedDict

from Utils.utils import choose_clients, fedavg

##LEARNING##

def federated_learning(model: BADFU, device, clients: list, avoid_client_id = None):
    curr_model = copy.deepcopy(model)

    for rnd in range(NUM_ROUNDS): 
        client_results = [] 
    
        clients_partecipating = choose_clients(clients=clients, avoid_client_id = avoid_client_id)

        for client in clients_partecipating :
            local_model = copy.deepcopy(curr_model) # each client has its own model to train locally 

            state_dict, n_samples, loss = client.train_model(local_model, device)

            client_results.append((state_dict, n_samples, loss))
        #tutti i client hanno fatto il loro training locale, quindi possiamo fare l'update del modello
        curr_model = fedavg(curr_model, client_results)

    return curr_model

##UNLEARNING##

def retraining_without_client(prev_model: BADFU, device, clients: list, client_id: int):
    retrained_model = federated_learning(prev_model, device, clients, client_id)
    return retrained_model





