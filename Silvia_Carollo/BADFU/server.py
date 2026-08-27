import copy 
import random 
import torch
from torch.utils.data import DataLoader , Subset

from collections import OrderedDict 
from config import TRIGGER_VAL, TRIGGER_SIZE, TARGET_LABEL

from model import BADFU
from Utils.Request import Request
from Utils.federated_unlearning import federated_unlearning
from Utils.federated_learning import federated_learning, retraining_without_client     

from config import BATCH_SIZE

class Server():
    def __init__(self, model: BADFU, test_data):
        self.prev_model = model
        self.model = model
        self.clients = []
        self.test_data = test_data

    ##CLIENTS##
    def add_client(self, client):
        self.clients.append(client)

    def get_bad_clients(self):
        from clients import BadClient
        return [client for client in self.clients if isinstance(client,BadClient)]

    
    ##LEARNING##
    def train(self, clients, device, avoid_client_id = None):
        self.prev_model = copy.deepcopy(self.model) 
        self.model = federated_learning(self.model, device, clients, avoid_client_id = None)

    ##UNLEARNING##

    def unlearn(self, device, client_id : int, samples_to_erase: Subset):
        #match type : 
        #   case "retraining":
        #        retraining_without_client(self.prev_model, client_id, device)
        #case "FedU":
        federated_unlearning(self.model, self.clients, client_id, samples_to_erase, device)
        #    case _:
        #        print(f"unsupported unlearning type: {type_unl}\n")


    def evaluate(self, device):
        """
        Calcola contemporaneamente:
        - Accuracy sul test set pulito
        - Attack Success Rate (ASR) sul test set con trigger
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

                poisoned_imgs = imgs.clone()

                poisoned_imgs[:, 0, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = TRIGGER_VAL

                poisoned_predictions = self.model(poisoned_imgs).argmax(1)

                successful += (
                    poisoned_predictions == TARGET_LABEL
                ).sum().item()

                total_backdoor += labels.size(0)

        self.model.to("cpu")

        accuracy = correct / total
        asr = successful / total_backdoor

        print(f'evaluation terminated.\nAccuracy: {accuracy*100:.2f}% \nASR: {asr*100:.2f}%')
        return accuracy, asr




