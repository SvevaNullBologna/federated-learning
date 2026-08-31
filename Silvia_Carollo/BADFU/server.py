import copy 
import random 
import torch
import wandb
from torch.utils.data import DataLoader , Subset

from collections import OrderedDict 
from config import TRIGGER_VAL, TRIGGER_SIZE, TARGET_LABEL

from model import BADFU
from Utils.utils import fedavg, choose_clients
from Utils.Request import Request
from Utils.federated_learning import federated_learning, retraining_without_client     

from config import BATCH_SIZE, CLIENTS_PART

class Server():
    def __init__(self, model: BADFU, test_data):
        self.model = model
        self.clients = []
        self.test_data = test_data
        self.requests = Request()

    ##CLIENTS##
    def add_client(self, client):
        self.clients.append(client)

    def get_bad_clients(self):#metodo per il test, ovviamente nella realtà il server non sa chi sono i client bizantini
        from clients import BadClient
        return [client for client in self.clients if isinstance(client,BadClient)]

    
    ##LEARNING##
    def train(self, clients, device, clients_part = CLIENTS_PART, clients_to_have = None, clients_to_avoid = None):
        self.model = federated_learning(self.model, device, clients, clients_part, clients_to_have, clients_to_avoid)

    ##UNLEARNING##

    def unlearning(self, model, device, clients_part = CLIENTS_PART):
        self.evaluate_clients(device, "pre_unlearning")

        results = []

        #####always_present = request.ids 
        always_present = self.requests.get_all_ids()

        unlearning_clients = choose_clients(self.clients, clients_part, always_present, None)

        for client in unlearning_clients:
            local_model = copy.deepcopy(model)
            results.append(client.unlearn_model(local_model, self.requests, device ))
        
        self.model = fedavg(model, results)

    def request_unlearning(self, client_id : int, samples_to_erase: Subset):
        self.requests.add(client_id, samples_to_erase)

    def evaluate_clients(self, device, stage:str="eval"):
        for client in self.clients:
            client.evaluate_forgetting(self.model, device, stage)
            client.evaluate_retain(self.model, device, stage )
  
    def evaluate(self, device, stage: str = "eval"):
        """
        Calcola contemporaneamente:
        - Accuracy sul test set pulito
        - Attack Success Rate (ASR) sul test set con trigger.
        //////Bisogna eliminare quelle che hanno la stessa target label!
        """

        loader = DataLoader(
            self.test_data,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        self.model.to(device).eval()

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

                outputs = self.model(imgs)
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

                    poisoned_predictions = self.model(poisoned_imgs).argmax(1)

                    successful += (
                        poisoned_predictions == TARGET_LABEL
                    ).sum().item()

                    total_backdoor += mask.sum().item()

        self.model.to("cpu")

        accuracy = correct / total
        asr = successful / total_backdoor if total_backdoor > 0 else 0.0 

        print(f'evaluation terminated.\nAccuracy: {accuracy*100:.2f}% \nASR: {asr*100:.2f}%')
        
        if wandb.run is not None:
            wandb.log({f"{stage}/accuracy":accuracy, f"{stage}/asr":asr})
        
        return accuracy, asr


    def compare_models(self, old_model, new_model):
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

        print(
            f"MODEL CHANGE | "
            f"||theta_post - theta_pre|| = {total_diff:.8f} | "
            f"||theta_pre|| = {total_old:.8f} | "
            f"ratio = {total_diff / (total_old + 1e-12):.8e}"
        )

