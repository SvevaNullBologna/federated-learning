
def train(self, global_model, clients, device):
        for rnd in range(NUM_ROUNDS): 
            client_results = [] 
        
            clients_partecipating = self.choose_clients()

            for client in clients_partecipating :
                local_model = copy.deepcopy(global_model) # each client has its own model to train locally 

                state_dict, n_samples, loss = client.train_model(local_model, device)

                client_results.append((state_dict, n_samples, loss))
            
            #tutti i client hanno fatto il loro training locale, quindi possiamo fare l'update del modello
            return self.fedavg(model, client_results)
            



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
