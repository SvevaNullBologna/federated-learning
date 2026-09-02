import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader

from config import BATCH_SIZE, TARGET_LABEL, TRIGGER_SIZE, TRIGGER_VAL

# utilities
def _evaluate_dataset(model, dataset, device):
    """ calcola loss e accuracy su un singolo dataset """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0) 

        loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return {
            "loss":loss,
            "accuracy":accuracy,
            "samples":total
        }
#########################

def evaluate_subset(model,kept_subset,erased_subset,device,stage):
    # valuta il modello sul:
        # forgetting/erased set 
        # retain/kept set

    model = model.to(device)
    model.eval()

    erased_metrics = _evaluate_dataset(model, erased_subset, device)
    kept_metrics = _evaluate_dataset(model, kept_subset, device)

    model.to("cpu")

    results = {
        "erased_loss": erased_metrics["loss"],
        "erased_accuracy": erased_metrics["accuracy"],
        "erased_samples": erased_metrics["samples"],

        "kept_loss": kept_metrics["loss"],
        "kept_accuracy": kept_metrics["accuracy"],
        "kept_samples": kept_metrics["samples"]
    }

    if wandb.run is not None:
        wandb.log({
            f"{stage}/erased_loss": results["erased_loss"],
            f"{stage}/erased_accuracy": results["erased_accuracy"],
            f"{stage}/kept_loss": results["kept_loss"],
            f"{stage}/kept_accuracy": results["kept_accuracy"],
        })

    return results 


def evaluate_accuracy_and_ASR(model,test_data, device, stage: str = "eval"):
        """
        Calcola contemporaneamente:
        - Accuracy sul test set pulito
        - Attack Success Rate (ASR) sul test set con trigger.
        //////Bisogna eliminare quelle che hanno la stessa target label!
        """

        loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

        model.to(device).eval()

        correct = 0
        total = 0

        successful = 0
        total_backdoor = 0

        with torch.no_grad():

            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                # ==================================================
                # 1. ACCURACY SU IMMAGINI PULITE
                # ==================================================

                outputs = model(imgs)
                predictions = outputs.argmax(1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # ==================================================
                # 2. ASR SU IMMAGINI CON TRIGGER
                # ==================================================

                mask = labels != TARGET_LABEL
                if mask.any():
                    poisoned_imgs = imgs[mask].clone()
                    poisoned_imgs[:, 0, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = TRIGGER_VAL

                    poisoned_predictions = model(poisoned_imgs).argmax(dim=1)

                    successful += (poisoned_predictions == TARGET_LABEL).sum().item()

                    total_backdoor += mask.sum().item()

        model.to("cpu")

        accuracy = correct / max(total,1)
        asr = successful / total_backdoor if total_backdoor > 0 else 0.0 
        
        if wandb.run is not None:
            wandb.log({f"{stage}/accuracy":accuracy, f"{stage}/asr":asr})
        
        return accuracy, asr


def compare_models(old_model, new_model):
    total_diff = 0.0
    total_old = 0.0

    old_params = dict(old_model.named_parameters())
    new_params = dict(new_model.named_parameters())

    for name in old_params:
        old_p = old_params[name].detach().float()
        new_p = new_params[name].detach().float()

        total_diff += torch.sum((new_p - old_p) ** 2).item()
        total_old += torch.sum(old_p ** 2).item()

    total_diff = total_diff ** 0.5
    total_old = total_old ** 0.5

    if wandb.run is not None:
        wandb.log({
            "model_change/absolute": total_diff,
            "model_change/pre_norm": total_old,
            "model_change/ratio": ratio
        })