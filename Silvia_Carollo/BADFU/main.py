from model import BADFU
from config import initial_config, NUM_CLIENTS, PERC_BAD_CLIENTS
from clients import Client, BadClient 
from Utils.data import load_mnist, partition_iid
from server import Server
from torch.utils.data import Subset

def main():
    #global load of data for simulation 
    good_train, good_test = load_mnist() 
    print("loaded mnist")
    #create model
    model = BADFU()
    print("istantiated global model")
    #create server 
    server = Server(model, good_test) 
    print("instantiated server")

    #create clients 
        #each client must have its own data 
        #WE slice for ALL the clients NUM_GOOD_CLIENTS + NUM_BAD_CLIENTS, and the bad clients will modify their slice as they please
    client_indices = partition_iid(good_train, NUM_CLIENTS) # { client id : indexes of data for client }
    print("instantiated clients")

    NUM_BAD_CLIENTS = int(NUM_CLIENTS * PERC_BAD_CLIENTS)
    NUM_GOOD_CLIENTS = NUM_CLIENTS - NUM_BAD_CLIENTS

    print("GOOD CLIENTS: ", NUM_GOOD_CLIENTS)
    print("BAD CLIENTS: ", NUM_BAD_CLIENTS)

    for i in range(NUM_CLIENTS):
        train_subset = Subset(good_train, client_indices[i])
        client_class = Client if i < NUM_GOOD_CLIENTS else BadClient 
        server.add_client(client_class(i, train_subset))
    print("partitioned data")

    device = initial_config()
    print("device : ", device)
    # train model with federated learning
    server.train(server.clients, device)
    
    # check how many corrects over total, ecc...
    server.evaluate(device)
    
    # federated unlearning 
    bad_clients = server.get_bad_clients()

    # federated unlearning requested by the bad client
    for client in bad_clients:
        client.request_unlearning(server, device)
        print(f'client {client.id} has requested unlearning\n')

    print(f'unlearning. Wait...\n')
    server.unlearning(server.model, device)

    print(f"unlearning completed.\n")

    # check metrics now 
    server.evaluate(device)
    


if __name__ == "__main__":
    main()

"""




#learning rate linear decay schedule
def lr_schedule(initial_lr, round_t, total_rounds):
    return initial_lr * (1.0 - (round_t - 1) / total_rounds)

#Pretraining
def pretrain_model(model, dataset, epochs, batch_size, lr, device):
    model.to(device).train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if ep % 10 == 0 or ep == 1:
            print(f"  Pretrain epoch {ep}/{epochs} | Loss: {ep_loss / len(loader):.4f}")

    model.to("cpu")
    return model
"""

