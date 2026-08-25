from model import BADFU 
from config import CLIENTS_PART, NUM_ROUNDS, NUM_CLIENTS, EPOCHS, LR

import copy
import random

from collections import OrderedDict

##UTILS##

def choose_clients(clients: list, always_present_client_id = None , avoid_client_id = None):
    if not clients : 
        print("Client List is empty")
        return [] 

    if always_present_client_id is not None and always_present_client_id == avoid_client_id :
        print("Client cannot be always present while being avoided")
        return []
    
    available_clients = clients 

    if avoid_client_id is not None :
        available_clients = [c for c in clients if getattr(c, "id", c) != avoid_client_id]            
    if not available_clients:
        print("No available clients to choose from")
        return []

    # CLIENTS_PART is a percentage -> ex. len(clients) * 0.2
    sample_size = max(1, int(len(available_clients) * CLIENTS_PART))

    if always_present_client_id is not None :
        always_present = next((c for c in clients if c.id == always_present_client_id), None)
        if always_present is None:
            print("the always-present client was not found")
            return []
        
        candidates = [c for c in available_clients if c.id != always_present_client_id] # togliamo il client da quelli che vogliamo selezionare randomicamente
        remaining = min(sample_size - 1, len(candidates))
        selected = random.sample(candidates, remaining)
        return [always_present] + selected 

    return random.sample(available_clients, sample_size) 

#FedAvg
def fedavg(global_model: BADFU, client_results):
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



def federated_unlearning(model: BADFU, clients:list, client_id: int, samples_to_erase: list, device):
    # re-initialize the global model and send it to all users
    unlearned_model = copy.deepcopy(model)

    clients_partecipating = choose_clients(clients = clients, always_present_client_id = client_id)

    client_results = []

    for client in clients_partecipating :
        local_model = copy.deepcopy(unlearned_model)

        if(client.id == client_id):#if it's the client that requested the unlearning

            state_dict, n_samples, loss = client.unlearn(local_model, samples_to_erase, device)
        
        else:
            state_dict, n_samples, loss = client.train_model(local_model, device)
        
        client_results.append((state_dict, n_samples, loss))

    unlearned_model = fedavg(unlearned_model, client_results)

    return unlearned_model

