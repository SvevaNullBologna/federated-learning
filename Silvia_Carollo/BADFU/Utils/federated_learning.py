from Agents.model import BADFU 
from config import CLIENTS_PART, TRAINING_NUM_ROUNDS, NUM_CLIENTS, TRAINING_LR

import copy
import random

from collections import OrderedDict

from Utils.utils import choose_clients, fedavg

##LEARNING##

def federated_learning(model: BADFU, device, clients: list, clients_part = CLIENTS_PART, clients_to_have:list = None, clients_to_avoid:list = None, training_num_rounds = TRAINING_NUM_ROUNDS):
    curr_model = copy.deepcopy(model)

    for rnd in range(training_num_rounds): 
        client_results = [] 
    
        clients_partecipating = choose_clients(clients, clients_part , clients_to_have, clients_to_avoid)
        print(f"[FEDERATED LEARNING]: clients partecipating in round {rnd}: {[c.id for c in clients_partecipating]}")
        for client in clients_partecipating :
            local_model = copy.deepcopy(curr_model) # each client has its own model to train locally 

            state_dict, n_samples, loss = client.train_model(local_model, device)

            client_results.append((state_dict, n_samples, loss))
        #tutti i client hanno fatto il loro training locale, quindi possiamo fare l'update del modello
        curr_model = fedavg(curr_model, client_results)

    return curr_model

##UNLEARNING##

def retraining_without_client(prev_model: BADFU, device, clients: list, clients_to_avoid: list, clients_to_have:list = None, clients_part = CLIENTS_PART):
    retrained_model = federated_learning(prev_model, device, clients, clients_part, clients_to_have, clients_to_avoid)
    return retrained_model





