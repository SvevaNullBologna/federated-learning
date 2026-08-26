import copy
import torch
import torch.nn as nn

from Utils.utils import choose_clients, fedavg
from torch.utils.data import DataLoader, Subset
from config import BATCH_SIZE, DAMPING, SCALE, DEPTH, FU_EPOCHS, LR_PRESERVE_FU, LR_UPDATE_FU



"""
/////////////////////////////UTILS//////////////////////////
"""
def flatten_gradients(model):
    return torch.cat([
        p.grad.reshape(-1)
        for p in model.parameters()
        if p.grad is not None
    ])

def calculate_gu(local_model, erased_data, device):
    criterion = nn.CrossEntropyLoss()

    erased_loader = DataLoader(erased_data, batch_size=BATCH_SIZE, shuffle=False)

    # vettore che conterrà il gradiente rispetto a TUTTI i parametri
    g_u = torch.zeros(sum(p.numel() for p in local_model.parameters() if p.requires_grad),device=device)

    local_model.train()

    for imgs, labels in erased_loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        local_model.zero_grad()

        print("calculate gu")
        print("MODEL DEVICE:", next(local_model.parameters()).device)
        print("DATA DEVICE:", imgs.device)
        output = local_model(imgs)
        loss = criterion(output, labels)

        loss.backward()

        g_u += flatten_gradients(local_model)

    g_u /= len(erased_loader)

    return g_u

def hessian_vector_product(model, loss, vector):
    params = [p for p in model.parameters() if p.requires_grad]

    first_grads = torch.autograd.grad(loss,params,create_graph=True, retain_graph=True)

    grad_vector = torch.cat([g.reshape(-1) for g in first_grads ])

    grad_vector_product = torch.sum(grad_vector * vector)

    hvp = torch.autograd.grad(grad_vector_product,params)

    return torch.cat([h.reshape(-1) for h in hvp])

def influence_approx_forgetting(local_model, erased_data, kept_data, device):
    
    criterion = nn.CrossEntropyLoss()
    
    # calcolo del gradiente della loss sui dati da dimenticare
    g_u = calculate_gu(local_model, erased_data, device) 

    #controlliamo che ci siano dei dati da mantenere, altrimenti non si può costruire una stima Hessiana sui dati da mantenere
    if kept_data is None or len(kept_data) == 0 : 
        return g_u

    #dataloader dei dati da mantenere
    kept_loader = DataLoader(kept_data, batch_size = BATCH_SIZE, shuffle = True)
    
    kept_iterator = iter(kept_loader)

    #inizializziamo la stima
    v = g_u.clone()

    #ora facciamo delle iterazioni per approssimare H^(-1) g_u. L'hessiana vera e propria sarebbe difficile da calcolare
    local_model.train() #Mettiamo il modello in modalità training
    
    for _ in range(DEPTH):
        print("growth check:",torch.norm(v).item())
        try:
            imgs, labels = next(kept_iterator)
        except StopIteration:
            kept_iterator = iter(kept_loader)
            imgs, labels = next(kept_iterator)

        imgs = imgs.to(device)
        labels = labels.to(device)

        local_model.zero_grad()

        print("influence approx forgetting")
        print("MODEL DEVICE:", next(local_model.parameters()).device)
        print("DATA DEVICE:", imgs.device)
        output = local_model(imgs)
        loss = criterion(output, labels)

        hv = hessian_vector_product(local_model, loss, v)

        with torch.no_grad():
            v = g_u + (1.0 - DAMPING) * v - hv/ SCALE
    
    return v


def update_parameters(local_model, influence_delta):

    offset = 0

    with torch.no_grad():

        for param in local_model.parameters():

            numel = param.numel()

            influence_param = influence_delta[
                offset : offset + numel
            ].view_as(param)

            param -= LR_UPDATE_FU * influence_param

            offset += numel


def utility_preservation(local_model, kept_data, device):
    """
    Fine-tuning leggero sui dati da MANTENERE per ripristinare la precisione (Utility Preservation).
    """
    if kept_data is None or len(kept_data) == 0:
        return

    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=LR_PRESERVE_FU, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(kept_data, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(FU_EPOCHS):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(imgs), labels)
            loss.backward()
            optimizer.step()

"""
//////////////////////FUNCTIONS TO USE////////////////////////
"""

def federated_unlearning(model: BADFU, clients:list, client_id: int, samples_to_erase: list, device):
    # re-initialize the global model and send it to all users
    unlearned_model = copy.deepcopy(model)

    clients_partecipating = choose_clients(clients = clients, always_present_client_id = client_id)

    client_results = []

    for client in clients_partecipating :
        local_model = copy.deepcopy(unlearned_model)

        if(client.id == client_id):#if it's the client that requested the unlearning

            state_dict, n_samples, loss = client.unlearn(local_model, samples_to_erase, device)
        
        else:
            state_dict, n_samples, loss = client.train_model(local_model, device)
        
        client_results.append((state_dict, n_samples, loss))

    unlearned_model = fedavg(unlearned_model, client_results)

    return unlearned_model