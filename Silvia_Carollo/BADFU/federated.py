import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset 
from collections import OrderedDict

"""
def client_update(model, dataset, indices, epochs, batch_size, lr, device):
    model = model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=True, num_workers=0)

    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

    model.to("cpu")
    return model.state_dict(), len(indices), total_loss / max(n_batches, 1)
"""
#FedAvg
def fedavg(global_model, client_results):
    total_n = sum(n for _, n, _ in client_results)
    agg = OrderedDict()
    for sd, n, _ in client_results:
        w = n / total_n
        for key in sd:
            val = w * sd[key].float()
            agg[key] = val if key not in agg else agg[key] + val
    global_model.load_state_dict(agg)
    return global_model


def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    model.to("cpu")
    return correct / total

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


