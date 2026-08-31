from Agents.model import BADFU
from config import (initial_config, NUM_CLIENTS, PERC_BAD_CLIENTS, NUM_ROUNDS,
                     LEARNING_EPOCHS, LR, BATCH_SIZE, N_FU_EPOCHS, IAF_FU_EPOCHS, LR_UPDATE_FU,
                     DEPTH, DAMPING, SCALE, MAX_D_NORM, CLEAN_IMG, POISON_IMG,
                     SAMPLES_TO_ERASE,TRAIN_CLIENTS_PART, UNLEARN_CLIENTS_PART)

from Agents.clients import Client, BadClient 
from Utils.data import load_mnist, partition_iid
from Agents.server import Server
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
            "normal_training_fu_epochs": N_FU_EPOCHS,
            "iaf_fu_epochs": IAF_FU_EPOCHS,
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

    # train model with federated learning
    server.train(server.clients, device, TRAIN_CLIENTS_PART) #training iniziale, partecipano tutti

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
        print(f'client {client.id} has requested unlearning\n')
        client.request_unlearning(server)

    print(f'unlearning. Wait...\n')
    server.unlearning(server.model, device, UNLEARN_CLIENTS_PART)

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


