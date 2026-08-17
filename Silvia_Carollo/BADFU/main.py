from model import BADFU
from config import NUM_GOOD_CLIENTS, NUM_BAD_CLIENTS
from client import Client, BadClient 
from data import load_mnist, partition_iid

def main():
    #global load of data for simulation 
    good_train, good_test = load_mnist() 

    #create model
    model = BADFU()
    #create server 
    server = Server(model, good_test) 

    #create clients 
        #each client must have its own data 
        #WE slice for ALL the clients NUM_GOOD_CLIENTS + NUM_BAD_CLIENTS, and the bad clients will modify their slice as they please
    total_clients = NUM_GOOD_CLIENTS + NUM_BAD_CLIENTS
    client_indices = partition_iid(good_train, total_clients) # { client id : indexes of data for client }

    for i in range(total_clients):
        train_subset = Subset(good_train, client_indices[i])
        client_class = Client if i < NUM_GOOD_CLIENTS else BadClient 
        server.add_client(client_class(i, model.state_dict(), train_subset))

    # train model with federated learning
    server.train()
    
    # check how many corrects over total, ecc...
    server.evaluate()

    # federated unlearning 
    # check metrics now 

    # train model with federated learning with BAD CLIENT 
    # check how many corrects over total, ecc...
    # federated unlearning requested by the bad client
    # check metrics now