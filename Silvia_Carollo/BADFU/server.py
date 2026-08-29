import copy 
import random 
import torch
from torch.utils.data import DataLoader , Subset

from collections import OrderedDict 
from config import TRIGGER_VAL, TRIGGER_SIZE, TARGET_LABEL

from model import BADFU
from Utils.utils import fedavg
from Utils.Request import Request
from Utils.federated_learning import federated_learning, retraining_without_client     

from config import BATCH_SIZE

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
    def train(self, clients, device, avoid_client_id = None):
        self.model = federated_learning(self.model, device, clients, avoid_client_id)

    ##UNLEARNING##

    def unlearning(self, model, device):
        results = []
        for client in self.clients:
            local_model = copy.deepcopy(model)
            results.append(client.unlearn_model(local_model, self.requests, device ))
        
        self.model = fedavg(model, results)

    def request_unlearning(self, client_id : int, samples_to_erase: Subset):
        self.requests.add(client_id, samples_to_erase)

    def evaluate(self, device):
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
        return accuracy, asr




