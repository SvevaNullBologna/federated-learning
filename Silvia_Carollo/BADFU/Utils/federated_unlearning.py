import copy
import torch
import torch.nn as nn

from itertools import cycle 

from torch.utils.data import DataLoader, Subset
from config import BATCH_SIZE, DAMPING, SCALE, DEPTH, FU_EPOCHS, LR_UPDATE_FU, EPS



def IAF_U(model, data, client_id: int, indexes_to_erase: list[int], device):
    model = model.to(device)
    n_total = len(data)
    m = len(indexes_to_erase)
    # dataset dei campioni da dimenticare
    erased_data = Subset(data, indexes_to_erase)
    # otteniamo il dataset dei campioni da mantenere
    erase_set = set(indexes_to_erase)
    
    retain_indices = [
        i for i in range(len(data))
        if i not in erase_set
    ]

    kept_data = Subset(data, retain_indices)

    #dataloaders 
    erased_loader = DataLoader(erased_data, batch_size = min(BATCH_SIZE, len(erased_data)), shuffle = True, drop_last = False)
    kept_loader = DataLoader(kept_data, batch_size = BATCH_SIZE, shuffle = True, drop_last = False)
    
    erased_batches = cycle(erased_loader)
    kept_batches = cycle(kept_loader)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0 

    for epoch in range(FU_EPOCHS):
        erased_imgs, erased_labels = next(erased_batches)
        kept_imgs, kept_labels = next(kept_batches)

        erased_imgs = erased_imgs.to(device)
        erased_labels = erased_labels.to(device)
        kept_imgs = kept_imgs.to(device)
        kept_labels = kept_labels.to(device)

        #calcolo dei gradienti 
        # influence approximation forgetting
        g_iaf = iaf_direction(model, criterion, erased_imgs, erased_labels, kept_imgs, kept_labels, n_total, m)

        # utility preservation loss
        g_up, up_loss = up_grad(model, criterion, kept_imgs, kept_labels)

        total_loss += up_loss

        d, lam, beta = difference(g_iaf, g_up)

        #solo per debug:
        
        print(f"[client {client_id}] |pseudo_grad IAF| = {torch.norm(g_iaf).item():.6f} | "
            f"|grad UP| = {torch.norm(g_up).item():.6f} | |d| = {torch.norm(d).item():.6f} | "
            f"lambda = {lam.item():.3f}")

        apply_update(model, d, LR_UPDATE_FU)

        #if verified(model):
        #  print(f"[client {client_id}] verifica superata all'epoca {epoch}, arresto unlearning")
        #  break
    avg_loss = total_loss / max(FU_EPOCHS, 1)
    
    return model , avg_loss


def verified(model):
    pass

def normal_training(model, data, device):
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()

    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle = True, drop_last = False)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_UPDATE_FU, momentum = 0.9, weight_decay = 5e-4)
    
    total_loss = 0.0
    n_batches = 0

    for _ in range(FU_EPOCHS):
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1 

    avg_loss = total_loss/max(n_batches, 1)
    return model, avg_loss
 

""" COMPUTING FUNCTIONS """


def iaf_direction(model, criterion, erased_imgs, erased_labels, kept_imgs, kept_labels, n_total, m):
    """
        Liaf = arg min_theta* 1/(n-m) * sum(loss(z,theta)) 

        attenzione, la minimizzazione della media va fatta fuori. 
        #si fa loss.backward() per ottenere il gradiente 
    """

    params = trainable_params(model)
    theta0 = flat([p.detach() for p in params])

    model.zero_grad()
    out = model(erased_imgs)
    loss_erased_sum = criterion(out, erased_labels) #* erased_imgs.size(0)
    grads_erased = torch.autograd.grad(loss_erased_sum, params, create_graph = True)
    v = flat(grads_erased).detach() * m # stima non distorta della somma su TUTTI gli m campioni

    t = min(DEPTH, kept_imgs.size(0)) # t campioni per la ricorsione LISSA
    remaining_samples = [(kept_imgs[i:i + 1], kept_labels[i:i + 1]) for i in range(t)]

    ihvp = estimate_inverse_hvp(model, criterion, remaining_samples, v)

    theta_target = theta0 + (1.0 / max (n_total-m,1))*ihvp # Da Eq 6 del paper

    pseudo_grad = theta0 - theta_target 
    return pseudo_grad 

    
def up_grad(model, criterion, kept_imgs, kept_labels):
    """
        L_up 
        theta* = arg min_theta l(D, theta) {(1/n) [sum_z_belong_D (l(z, theta)) ]}
    """
    params = trainable_params(model)
    model.zero_grad() 
    out = model(kept_imgs)
    loss = criterion(out, kept_labels)
    grads = torch.autograd.grad(loss, params)
    return flat(grads).detach(), loss.item()



def difference(grad_iaf, grad_up):
    # d_k_u,t = - (lambda * gradient * Liaf(theta, t) + beta * gradient * Lup(theta,t)) where lambda + beta = 1 , labda >= 0, beta >= 0 
    g1 = grad_iaf 
    g2 = grad_up 
    
    diff = g1 - g2 
    denominator = torch.sum(diff * diff)
    
    if denominator.item() < EPS:
        lambda_ = torch.tensor(0.5, device=g1.device)
    else:
        lambda_ = torch.sum(g2 * (g2-g1)) / denominator 
        lambda_ = torch.clamp( lambda_, min=0.0, max=1.0)

    beta = 1.0 - lambda_
    d = -(lambda_ * g1 + beta * g2)

    return d, lambda_, beta


""" HESIAN VECTOR TO ESTIMATE """

def hessian_vector_product(loss, params, v):
    """ hessiana di loss rispetto a params """   
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_dot_v = torch.sum(flat(grads) * v)
    hvp = torch.autograd.grad(grad_dot_v, params, retain_graph=False)
    return flat(hvp).detach()

def estimate_inverse_hvp(model,criterion,remaining_samples, v, damping=DAMPING, scale=SCALE, repeats=1):
    """
    H~^-1_0 gu = gu
    H~^-1_i gu = gu + (I - grad^2 l(z_i, theta*)) H~^-1_{i-1} gu"""
    
    params = trainable_params(model)
    estimates = []

    for _ in range(repeats):
        cur = v.clone()
        for x, y in remaining_samples:
            model.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            hvp = hessian_vector_product(loss, params, cur)
            cur = v + (1 - damping) * cur - hvp / scale
        estimates.append(cur)
 
    return torch.stack(estimates, dim=0).mean(dim=0)

"""
//////////////////UTILS/////////////////
"""

def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]
 
 
def flat(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])
 
 
def unflatten_like(vec, params):
    out, idx = [], 0
    for p in params:
        n = p.numel()
        out.append(vec[idx:idx + n].view_as(p))
        idx += n
    return out
 
 
def apply_update(model, d_flat, lr):
    """theta = theta + lr * d  (aggiornamento in-place sui parametri del modello)."""
    params = trainable_params(model)
    d_list = unflatten_like(d_flat.detach(), params)
    with torch.no_grad():
        for p, d in zip(params, d_list):
            p.add_(lr * d)
 
 