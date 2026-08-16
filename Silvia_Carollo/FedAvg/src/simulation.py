import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import random
import copy
import torchvision.transforms as transforms
import wandb

from model import NeuralNetwork
from data import dataset_partition_IID, dataset_partition_not_IID
from client import client_update
from server import federated_averaging
from evaluation import evaluate


#simulazione algoritmo completo
def run_federated_averaging(num_clients = 2, fraction = 1, num_rounds = 50, local_epochs = 5, batch_size = 10, lr = 0.01, iid = True):

    #impostazione WandB

    wandb.init(project="FedAvg", config={
        "num_clients": num_clients,
        "fraction":fraction,
        "num_rounds":num_rounds,
        "local_epochs":local_epochs,
        "batch_size": batch_size,
        "lr":lr,
        "iid":iid,
    })

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.FashionMNIST(root="data",
                                          train = True,
                                          download = True,
                                          transform = transform,)

    test_dataset = datasets.FashionMNIST(root="data",
                                          train = False,
                                          download = True,
                                          transform = transform,)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


    if iid:
        client_data = dataset_partition_IID(train_dataset, num_clients)
    else:
        client_data = dataset_partition_not_IID(train_dataset, num_clients)


    global_model = NeuralNetwork()

    m = max(1, int(fraction*num_clients))

    for round_t in range(1, num_rounds+1):
        selected_clients = random.sample(range(num_clients), m)

        client_updates = []
        client_losses = []
        for k in selected_clients:
            local_model = copy.deepcopy(global_model)

            updated_weights, n_k, loss = client_update(client_model=local_model,
                                                 dataset=train_dataset,
                                                 indices=client_data[k],
                                                 epochs=local_epochs,
                                                 batch_size=batch_size,
                                                 lr=lr,
                                                 )
            client_updates.append((updated_weights, n_k))
            client_losses.append(loss)

        avg_round_loss = sum(client_losses) / len(client_losses)
        global_model = federated_averaging(global_model, client_updates)


        accuracy = evaluate(global_model, test_loader)
        wandb.log({"round": round_t, "accuracy":accuracy, "loss":avg_round_loss})

        if round_t % 5==0 or round_t ==1:
            print(f"Round {round_t:3d}/{num_rounds} | Loss: {avg_round_loss:.4f} | Accuratezza test: {accuracy:.4f} ({accuracy*100:.2f}%)")

    final_accuracy = evaluate(global_model, test_loader)
    wandb.log({"final_accuracy" : final_accuracy})
    print(f"\n  Accuratezza finale: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

    wandb.finish()
    return global_model
