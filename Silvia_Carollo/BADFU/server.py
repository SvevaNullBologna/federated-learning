from model import BADFU


class Server():
    def __init__(self, model: BADFU, test_data):
        self.model = model
        self.clients = []
        self.test_data = test_data

    def add_client(self, client):
        self.clients.append(client)

    def get_data(self, client_id, parameters):#each client has its own data, so the server can get it from the client
        pass 

    def train(self):
        # Training logic for the server
        pass

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