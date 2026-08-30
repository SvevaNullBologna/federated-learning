from model import BADFU
from config import (initial_config, NUM_CLIENTS, PERC_BAD_CLIENTS, NUM_ROUNDS,
                     LEARNING_EPOCHS, LR, BATCH_SIZE, FU_EPOCHS, LR_UPDATE_FU,
                     DEPTH, DAMPING, SCALE, MAX_D_NORM, CLEAN_IMG, POISON_IMG,
                     SAMPLES_TO_ERASE, CLIENTS_PART)

from clients import Client, BadClient 
from Utils.data import load_mnist, partition_iid
from server import Server
from torch.utils.data import Subset

import copy

import wandb

def main():
    """
    ////////////////////////////////////////////////////////////////////////////////////////
                    INITIALIZATION 
    ////////////////////////////////////////////////////////////////////////////////////////
    """
    wandb.init(project="badfu", config={
        "num_clients": NUM_CLIENTS,
            "perc_bad_clients": PERC_BAD_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "learning_epochs": LEARNING_EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "fu_epochs": FU_EPOCHS,
            "lr_update_fu": LR_UPDATE_FU,
            "depth": DEPTH,
            "damping": DAMPING,
            "scale": SCALE,
            "max_d_norm": MAX_D_NORM,
            "clean_img": CLEAN_IMG,
            "poison_img": POISON_IMG,
            "samples_to_erase": SAMPLES_TO_ERASE
    })
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

    """
    ////////////////////////////////////////////////////////////////////////////////////////
                    TRAINING 
    ////////////////////////////////////////////////////////////////////////////////////////
    """

    clients_part = 1.0 if NUM_CLIENTS <= 10 else CLIENTS_PART  

    # train model with federated learning
    server.train(server.clients, device, clients_part) #training iniziale, partecipano tutti

    old_model = copy.deepcopy(server.model)#to check later
    
    # check how many corrects over total, ecc...
    server.evaluate(device, "pre_unlearning")

    """
    ////////////////////////////////////////////////////////////////////////////////////////
                    UNLEARNING 
    ////////////////////////////////////////////////////////////////////////////////////////
    """

    # federated unlearning 
    bad_clients = server.get_bad_clients()
    for client in bad_clients:
        client.request_unlearning(server)

    # federated unlearning requested by the bad client
    #for client in bad_clients:
     #   client.request_unlearning(server, device)
    #   print(f'client {client.id} has requested unlearning\n')

    print(f'unlearning. Wait...\n')
    server.unlearning(server.model, device)

    print(f"unlearning completed.\n")

    """
    ////////////////////////////////////////////////////////////////////////////////////////
                    METRICS CHECK 
    ////////////////////////////////////////////////////////////////////////////////////////
    """

    # check metrics now 
    server.evaluate(device, "post_unlearning")
    server.evaluate_clients(device, "post_unlearning")
    server.compare_models(old_model, server.model)

    wandb.finish()


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

