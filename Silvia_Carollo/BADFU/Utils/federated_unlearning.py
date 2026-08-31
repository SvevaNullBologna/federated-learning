import copy
import torch
import torch.nn as nn
import wandb

from itertools import cycle

from torch.utils.data import DataLoader, Subset
from config import BATCH_SIZE, DAMPING, SCALE, DEPTH, IAF_FU_EPOCHS, N_FU_EPOCHS, LR_UPDATE_FU, EPS, LISSA_BATCH_SIZE, MAX_D_NORM, IAF_SCALE, FIXED_LAMBDA


def IAF_U(model, data, client_id: int, indexes_to_erase: list, device):
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

    # dataloaders
    # per gli erased usiamo TUTTI gli m campioni in un solo batch: ci serve
    # la somma ESATTA dei gradienti (Eq. 6), non una stima rumorosa estrapolata
    # da un sotto-campione (che con m grande amplifica troppo il rumore)
    erased_loader = DataLoader(erased_data, batch_size=len(erased_data), shuffle=True, drop_last=False)
    kept_loader = DataLoader(kept_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    kept_batches = cycle(kept_loader)

    criterion = nn.CrossEntropyLoss()

    # --- direzione IAF: calcolata UNA SOLA VOLTA, al punto originale theta* ---
    # Eq. 6 e' una correzione valida localmente, vicino al punto in cui e'
    # stata stimata l'Hessiana. Ricalcolarla ad ogni epoca dopo che il modello
    # si e' gia' spostato la rende sempre meno affidabile: il valore cresce
    # epoca dopo epoca (l'abbiamo visto nei log) e il combinatore MGDA finisce
    # per darle peso ~0 perche' la giudica troppo "in conflitto" con grad_UP.
    # Congelandola qui evitiamo la deriva, e risparmiamo anche il costo di
    # ristimare l'Hessiana ad ogni epoca.
    erased_imgs, erased_labels = next(iter(erased_loader))
    erased_imgs = erased_imgs.to(device)
    erased_labels = erased_labels.to(device)

    kept_imgs0, kept_labels0 = next(kept_batches)
    kept_imgs0 = kept_imgs0.to(device)
    kept_labels0 = kept_labels0.to(device)

    g_iaf = iaf_direction(model, criterion, erased_imgs, erased_labels, kept_imgs0, kept_labels0, n_total, m)
    print(f"[client {client_id}] |pseudo_grad IAF| (fissa) = {torch.norm(g_iaf).item():.6f}")
    if wandb.run is not None:
        wandb.log({f"unlearning/client_{client_id}/iaf_norm_fixed": torch.norm(g_iaf).item()})

    total_loss = 0.0

    for epoch in range(IAF_FU_EPOCHS):
        kept_imgs, kept_labels = next(kept_batches)
        kept_imgs = kept_imgs.to(device)
        kept_labels = kept_labels.to(device)

        # utility preservation loss (questa si ricalcola ogni epoca: e' un
        # gradiente di training normale, stabile per costruzione)
        g_up, up_loss = up_grad(model, criterion, kept_imgs, kept_labels)

        total_loss += up_loss

        d, lam, beta = fixed_combine(g_iaf, g_up, FIXED_LAMBDA)  # niente più MGDA: peso garantito a g_iaf

        # clip sulla norma di d: rete di sicurezza aggiuntiva
        d_norm = torch.norm(d)
        if d_norm > MAX_D_NORM:
            d = d * (MAX_D_NORM / (d_norm + 1e-12))

        # solo per debug: stampa/logga la prima, l'ultima, e una ogni 10 epoche
        # (loggare su wandb OGNI epoca con FU_EPOCHS alto e' quello che stava
        # rallentando tutto: ogni wandb.log() e' una chiamata di rete, con
        # FU_EPOCHS=300 su 2 client erano 600 chiamate)
        if epoch == 0 or epoch == IAF_FU_EPOCHS - 1 or epoch % 10 == 0:
            print(f"[client {client_id}] epoch {epoch} | |grad UP| = {torch.norm(g_up).item():.6f} | "
                  f"|d| = {torch.norm(d).item():.6f} | lambda = {lam.item():.3f}")

            if wandb.run is not None:
                wandb.log({
                    f"unlearning/client_{client_id}/epoch": epoch,
                    f"unlearning/client_{client_id}/up_norm": torch.norm(g_up).item(),
                    f"unlearning/client_{client_id}/d_norm": torch.norm(d).item(),
                    f"unlearning/client_{client_id}/lambda": lam.item(),
                    f"unlearning/client_{client_id}/up_loss": up_loss,
                })

        apply_update(model, d, LR_UPDATE_FU)

        # if verified(model):
        #   print(f"[client {client_id}] verifica superata all'epoca {epoch}, arresto unlearning")
        #   break

    avg_loss = total_loss / max(IAF_FU_EPOCHS, 1)

    return model, avg_loss


def verified(model):
    pass


def normal_training(model, data, device):
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_UPDATE_FU, momentum=0.9, weight_decay=5e-4)

    total_loss = 0.0
    n_batches = 0

    for epoch in range(N_FU_EPOCHS):
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
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
    # erased_imgs ora contiene TUTTI gli m campioni cancellati (vedi DataLoader
    # in IAF_U), quindi mean*count e' gia' la somma ESATTA richiesta dall'Eq. 6
    # -- niente piu' bisogno di estrapolare moltiplicando per m
    loss_erased_sum = criterion(out, erased_labels) * erased_imgs.size(0)
    grads_erased = torch.autograd.grad(loss_erased_sum, params, create_graph=True)
    v = flat(grads_erased).detach()

    t = min(DEPTH, kept_imgs.size(0))  # totale campioni usati per la ricorsione LISSA
    lissa_batch_size = min(LISSA_BATCH_SIZE, t)  # quanti campioni per passo di ricorsione
    remaining_samples = [
        (kept_imgs[i:i + lissa_batch_size], kept_labels[i:i + lissa_batch_size])
        for i in range(0, t, lissa_batch_size)
    ]
    # es. t=20, lissa_batch_size=5 -> 4 passi di ricorsione invece di 20,
    # stessa informazione (20 campioni), 5x meno forward/backward

    ihvp = estimate_inverse_hvp(model, criterion, remaining_samples, v)

    theta_target = theta0 + IAF_SCALE * ihvp  # scala trattata come iperparametro, non più 1/(n-m) fisso

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
        lambda_ = torch.sum(g2 * (g2 - g1)) / denominator
        lambda_ = torch.clamp(lambda_, min=0.0, max=1.0)

    beta = 1.0 - lambda_
    d = -(lambda_ * g1 + beta * g2)

    return d, lambda_, beta


def fixed_combine(grad_iaf, grad_up, lam=0.5):
    """
    Combinazione a pesi FISSI, non adattiva. Usata dal paper BadFU stesso
    quando testano contro FedU (lambda=0.5, beta=0.5): a differenza di
    difference() (MGDA), qui g_iaf ottiene sempre un peso garantito,
    indipendentemente da quanto sia "in conflitto" con g_up in termini di
    magnitudo. Con MGDA, aumentare artificialmente la scala di g_iaf viene
    automaticamente compensato da un lambda che scende verso 0 - qui no.
    """
    g1 = grad_iaf
    g2 = grad_up
    beta = 1.0 - lam
    d = -(lam * g1 + beta * g2)
    return d, torch.tensor(lam, device=g1.device), torch.tensor(beta, device=g1.device)


""" HESIAN VECTOR TO ESTIMATE """


def hessian_vector_product(loss, params, v):
    """ hessiana di loss rispetto a params """
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_dot_v = torch.sum(flat(grads) * v)
    hvp = torch.autograd.grad(grad_dot_v, params, retain_graph=False)
    return flat(hvp).detach()


def estimate_inverse_hvp(model, criterion, remaining_samples, v, damping=DAMPING, scale=SCALE, repeats=1):
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