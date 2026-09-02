import torch
import numpy as np
import random
from enum import Enum 

def initial_config():
    #seed set for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    #Using mainly Mac for training
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


#Hyperparameters following the setting of the paper
#FL training (paper Section VI)
NUM_CLIENTS = 5
PERC_BAD_CLIENTS = 0.5

TRAIN_CLIENTS_PART = 1.0
UNLEARN_CLIENTS_PART = 0.5

CLEAN_IMG = 0.85
POISON_IMG = 0.08 

TARGET_LABEL = 0
TRIGGER_SIZE = 3
TRIGGER_VAL = (1.0 - 0.1307) / 0.3081

LEARNING_EPOCHS = 3
BATCH_SIZE = 32 
TRAINING_LR = 0.01    #paper: "learning rate is set to 0.01"

TRAINING_NUM_ROUNDS = 2 #paper: "80 training epochs"
CLIENTS_PART = 0.2 #20% client participation per round 

NUM_CLASSES = 10 

# PARAMETRI LISSA #
FU_DEPTH = 20 #to figure out how much to approx the hessian
FU_DAMPING = 0.1
FU_SCALE = 100.0

IAF_FU_EPOCHS = 15
N_FU_EPOCHS = 3
FIXED_LAMBDA = 0.5 
IAF_SCALE = 100

LR_UPDATE_FU = 0.05

EPS = 1e-12

LISSA_BATCH_SIZE = 5
MAX_D_NORM = 2.0
  
SAMPLES_TO_ERASE = 0.1

        #full MNIST (instead of only classes 0 and 1)
DATA_PER_CLIENT  = 1          #paper Sec VI.B: "set the number of data points per client to 1"

TYPE_OF_CLIENTS_WHO_REQUEST_UNLEARNING = 1 # 0 = BAD_CLIENTS, 1 = GOOD_CLIENTS, 2 = ALL_CLIENTS

