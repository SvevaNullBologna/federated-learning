import random
from collections import OrderedDict
from config import CLIENTS_PART

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
def fedavg(global_model, client_results):
    #quanti campioni in totale provenienti da TUTTI i client?
    total_n = sum(n for _, n, _ in client_results) # client_results possiede triple del tipo (state_dict, n_samples, loss). Stiamo sommando gli n_samples
    
    agg = OrderedDict() #creiamo una directory ordinata 

    for sd, n, _ in client_results: #cicliamo sulle tuple di client_results e prendiamo state_dict ed n_samples 
        w = n / total_n # calcoliamo il valore del peso (basato su quanti samples rispetto a quelli complessivi)
        for key in sd:
            val = w * sd[key].float().cpu() #calcoliamo il valore del peso in base agli state_dict 
            agg[key] = val if key not in agg else agg[key] + val

    global_model.load_state_dict(agg)

    return global_model
