from Agents.model import BADFU
from config import (initial_config, NUM_CLIENTS, PERC_BAD_CLIENTS, TRAINING_NUM_ROUNDS,
                     LEARNING_EPOCHS, TRAINING_LR, BATCH_SIZE, N_FU_EPOCHS, IAF_SCALE, IAF_FU_EPOCHS, LR_UPDATE_FU,
                     FU_DEPTH, FU_DAMPING, FU_SCALE, MAX_D_NORM, CLEAN_IMG, POISON_IMG,
                     SAMPLES_TO_ERASE,TRAIN_CLIENTS_PART, UNLEARN_CLIENTS_PART)

from Agents.clients import Client, BadClient 
from Agents.server import Server

from Utils.data import load_mnist, partition_iid
from Utils.utils import get_learning_clients_percentage

from torch.utils.data import Subset

import copy

import wandb

import Utils.evaluation as valuate

def main():
    """
    ////////////////////////////////////////////////////////////////////////////////////////
                    INITIALIZATION 
    ////////////////////////////////////////////////////////////////////////////////////////
    """
    config = {
        "num_clients": NUM_CLIENTS,
        "perc_bad_clients": PERC_BAD_CLIENTS,

        "learning_num_rounds": TRAINING_NUM_ROUNDS,
        "learning_epochs": LEARNING_EPOCHS,
        "training_lr": TRAINING_LR,

        "normal_training_fu_epochs": N_FU_EPOCHS,
        "iaf_fu_epochs": IAF_FU_EPOCHS,
        "iaf_scale": IAF_SCALE,
        "lr_update_fu": LR_UPDATE_FU,

        "fu_depth": FU_DEPTH,
        "fu_damping": FU_DAMPING,
        "fu_scale": FU_SCALE,
        "max_d_norm": MAX_D_NORM,

        "clean_img_per": CLEAN_IMG,
        "poison_img_per": POISON_IMG,

        "samples_to_erase": SAMPLES_TO_ERASE,
        "unlearn_clients_part": UNLEARN_CLIENTS_PART
    }

    with wandb.init(project="badfu", config=config) as run:

        cfg = wandb.config

        print("\n==============================")
        print("EXPERIMENT CONFIGURATION")
        print("==============================")
        print(f"NUM_CLIENTS          : {cfg.num_clients}")
        print(f"PERC_BAD_CLIENTS     : {cfg.perc_bad_clients}")
        print(f"IAF_FU_EPOCHS        : {cfg.iaf_fu_epochs}")
        print(f"POISON_IMG            : {cfg.poison_img_per}")
        print(f"CLEAN_IMG             : {cfg.clean_img_per}")
        print(f"UNLEARN_CLIENTS_PART  : {cfg.unlearn_clients_part}")
        print("==============================\n")

        ######## INITIALIZATION

        #global load of data for simulation 
        good_train, good_test = load_mnist() 
        print("loaded mnist")

        #create model
        model = BADFU()
        print("istantiated global model")


        #create server 
        server = Server(model, good_test) 
        print("instantiated server")

        
        ######### CLIENT PARTITION

        #create clients 
            #each client must have its own data 
            #WE slice for ALL the clients NUM_GOOD_CLIENTS + NUM_BAD_CLIENTS, and the bad clients will modify their slice as they please
        
        client_indices = partition_iid(good_train, cfg.num_clients) # { client id : indexes of data for client }
        print("instantiated clients")

        num_bad_clients = int(cfg.num_clients * cfg.perc_bad_clients)
        num_good_clients = cfg.num_clients - num_bad_clients

        print("GOOD CLIENTS: ", num_good_clients)
        print("BAD CLIENTS: ", num_bad_clients)

        for i in range(cfg.num_clients):
            train_subset = Subset(good_train, client_indices[i])
            client_class = Client if i < num_good_clients else BadClient 
            server.add_client(client_class(i, train_subset))
        print("partitioned data")

        device = initial_config()
        print("device : ", device)

        """
        ////////////////////////////////////////////////////////////////////////////////////////
                        TRAINING 
        ////////////////////////////////////////////////////////////////////////////////////////
        """

        print("\n========== TRAINING ==========\n")

        train_clients_part = get_learning_clients_percentage(cfg.num_clients)  
        # train model with federated learning
        
        print(
            f"Training client participation: "
            f"{train_clients_part*100:.1f}%"
        )
        
        server.train(server.clients, device, train_clients_part) #training iniziale, partecipano tutti

        """
        ////////////////////////////////////////////////////////////////////////////////////////
                        PRE-UNLEARNING 
        ////////////////////////////////////////////////////////////////////////////////////////
        """

        print("\n========== PRE-UNLEARNING ==========\n")

        #saving pre-unlearning model to get metrics later
        old_model = copy.deepcopy(server.model)#to check later
        
        # check how many corrects over total, ecc...
        evaluate_accuracy_and_ASR(server.model, server.data, device,stage="pre_unlearning")        

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

        print("\n========== UNLEARNING ==========\n")
        server.unlearning(server.model, device, cfg.unlearn_clients_part) #unlearning, partecipano solo i client che hanno richiesto l'unlearning

        print(f"unlearning completed.\n")

        """
        ////////////////////////////////////////////////////////////////////////////////////////
                        METRICS CHECK 
        ////////////////////////////////////////////////////////////////////////////////////////
        """
        print("\n========== POST-UNLEARNING ==========\n")

        # check metrics now 
        evaluate_accuracy_and_ASR(server.model, server.data, device,stage="post_unlearning")
        valuate.compare_models(old_model, server.model)

         # ========================================================
        # DERIVED METRICS
        # ========================================================

        pre_acc = run.summary.get(
            "pre_unlearning/accuracy"
        )

        post_acc = run.summary.get(
            "post_unlearning/accuracy"
        )

        pre_asr = run.summary.get(
            "pre_unlearning/asr"
        )

        post_asr = run.summary.get(
            "post_unlearning/asr"
        )

        if pre_acc is not None and post_acc is not None:
            wandb.summary["accuracy_drop"] = (
                pre_acc - post_acc
            )

        if pre_asr is not None and post_asr is not None:
            wandb.summary["asr_increase"] = (
                post_asr - pre_asr
            )

        # ========================================================
        # CAMOUFLAGE / POISON RATIO
        # ========================================================

        camouflage_img_per = (
            1.0
            - cfg.clean_img_per
            - cfg.poison_img_per
        )

        wandb.summary["camouflage_img_per"] = (
            camouflage_img_per
        )

        if cfg.poison_img_per > 0:
             wandb.summary["camouflage_poison_ratio"] = (
                camouflage_img_per
                / cfg.poison_img_per
            )



if __name__ == "__main__":
    main()


