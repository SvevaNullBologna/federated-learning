import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import time
import wandb

from config import (NUM_CLIENTS, FRACTION, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
                    FL_LR, PRETRAIN_EPOCHS, PRETRAIN_LR, DATA_PER_CLIENT, DEVICE)
from model import BADFU
from data import load_mnist, partition_iid
from federated import client_update, fedavg, evaluate, lr_schedule, pretrain_model



def run_fl_training():
    train_data, test_data = load_mnist()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    #80% - 20% split
    n_total = len(train_data)
    n_pretrain = int(0.8 * n_total)
    pretrain_data, private_data = torch.utils.data.random_split(
        train_data, [n_pretrain, n_total - n_pretrain])

    #Pre-train
    model = BADFU()
    print(f"Pre-training on {n_pretrain} samples ({PRETRAIN_EPOCHS} epochs)...")
    t0 = time.time()
    model = pretrain_model(model, pretrain_data, PRETRAIN_EPOCHS, BATCH_SIZE, PRETRAIN_LR, DEVICE)
    acc = evaluate(model, test_loader, DEVICE)
    print(f"  Done ({time.time() - t0:.0f}s) | Accuracy: {acc * 100:.1f}%\n")
    pretrained_sd = copy.deepcopy(model.state_dict())

    #Partition: 1 sample per client
    client_data = partition_iid(private_data, NUM_CLIENTS, DATA_PER_CLIENT)

    #FL rounds
    m = max(1, int(FRACTION * NUM_CLIENTS))
    stored_updates = {}   #(round, client) -> param_diff dict
    round_selections = {} #round -> list of selected clients

    print(f"FL Training: {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients, "
          f"{m} selected/round, LR={FL_LR}")
    t0 = time.time()

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = random.sample(range(NUM_CLIENTS), m)
        round_selections[rnd] = selected
        global_sd = copy.deepcopy(model.state_dict())
        lr = lr_schedule(FL_LR, rnd, NUM_ROUNDS)

        results = []
        for k in selected:
            local = copy.deepcopy(model)
            sd, n_k, loss = client_update(local, private_data, client_data[k],
                                          LOCAL_EPOCHS, BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
            stored_updates[(rnd, k)] = {
                key: sd[key].float() - global_sd[key].float() for key in sd
            }

        model = fedavg(model, results)

        if rnd % 10 == 0 or rnd == 1:
            acc = evaluate(model, test_loader, DEVICE)
            avg_loss = np.mean([r[2] for r in results])
            print(f"  Round {rnd:3d}/{NUM_ROUNDS} | LR: {lr:.5f} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc * 100:.1f}%")
            wandb.log({"fl/round": rnd, "fl/accuracy": acc,
                       "fl/loss": avg_loss, "fl/lr": lr})

    fl_time = time.time() - t0
    final_acc = evaluate(model, test_loader, DEVICE)
    print(f"  FL done ({fl_time:.0f}s) | Final accuracy: {final_acc * 100:.1f}%")
    wandb.log({"fl/final_accuracy": final_acc, "fl/time_s": fl_time})

    return (model, stored_updates, client_data, private_data,
            pretrained_sd, round_selections, test_loader)