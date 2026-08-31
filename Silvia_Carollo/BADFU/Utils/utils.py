import random
from collections import OrderedDict

##UTILS##

def choose_clients(clients: list, clients_part = 0.2, always_present:list = None , to_avoid:list = None) -> list:
    if not clients : 
        print("Client List is empty")
        return [] 

    always_set = set(always_present) if always_present else set()
    avoid_set = set(to_avoid) if to_avoid else set()

    if not always_set.isdisjoint(avoid_set):
        print("Client cannot be always present while being avoided")
        return []
    
    client_map = {getattr(c, "id", c): c for c in clients}

    #controllo preliminare per vedere se i client che DEVONO essere presenti, possono essere scelti 
    always_present_clients = [client_map[c_id] for c_id in always_set if c_id in client_map]
    
    if len(always_present_clients) < len(always_set):
        print("Warning: One or more always-present clients were not found")
        return []

    # prendiamo solo i candidati disponibili, senza quelli da escludere
    exclude_set = avoid_set.union(always_set)
    candidates = [c for c in clients if getattr(c, "id", c) not in exclude_set]

    #calcolo della dimensione del campione
    total_available_count = len(always_present_clients) + len(candidates)
    if total_available_count == 0:
        print("No available clients to choose from")
        return []

    # CLIENTS_PART is a percentage -> ex. len(clients) * 0.2
    sample_size = max(1, int(total_available_count * clients_part))

    remaining_needed = sample_size - len(always_present_clients)

    if remaining_needed > 0:
        actual_sample_count = min(remaining_needed, len(candidates))
        selected_random = random.sample(candidates, actual_sample_count)
        return always_present_clients + selected_random
    else:
        return always_present_clients

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



class Request():
    def __init__(self):
        self.requests = []

    def add(self, id: int, indexes_to_erase: list[int] = None):
        if indexes_to_erase is None:
            print(f"no indexes to erase, the request is useless\n")
            return 
        self.requests.append((id, indexes_to_erase))
    
    def remove(self, id: int):
        self.requests = [req for req in self.requests if req[0] != id]

    def contains(self, id: int) -> bool:
        return any(req[0] == id for req in self.requests)

    def get(self, id: int):
        for req in self.requests:
            if req[0] == id:
                return req 
        return None 

    def get_all_ids(self) -> list[int]:
        return [req[0] for req in self.requests]