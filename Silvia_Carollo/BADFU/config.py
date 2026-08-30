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

CLEAN_IMG = 0.85
POISON_IMG = 0.08 

TARGET_LABEL = 0 
TRIGGER_SIZE = 3
TRIGGER_VAL = (1.0 - 0.1307) / 0.3081

LEARNING_EPOCHS = 3
BATCH_SIZE = 32 
LR = 0.01    #paper: "learning rate is set to 0.01"

NUM_ROUNDS = 2 #paper: "80 training epochs"
CLIENTS_PART = 0.2 #20% client participation per round 

NUM_CLASSES      = 10 

# PARAMETRI LISSA #
DEPTH = 20 #to figure out how much to approx the hessian
DAMPING = 0.1
SCALE = 100.0

IAF_FU_EPOCHS = 15
N_FU_EPOCHS = 3

LR_UPDATE_FU = 0.05

EPS = 1e-12

LISSA_BATCH_SIZE = 5
MAX_D_NORM = 2.0


FL_LR            = 0.01      
PRETRAIN_EPOCHS  = 5          #the paper specify 50
PRETRAIN_LR      = 0.01

SAMPLES_TO_ERASE = 0.1

        #full MNIST (instead of only classes 0 and 1)
DATA_PER_CLIENT  = 1          #paper Sec VI.B: "set the number of data points per client to 1"

FIXED_LAMBDA = 0.5 
FIXED_BETA = 0.5 

IAF_SCALE = 0.1

# Gradient inversion
INV_ITERATIONS   = 20000
INV_LR           = 0.1
INV_GAMMA        = 0.1        #paper-faithful: weight of Psi term in Eq. 18
INV_ALPHA        = 1e-5       #minimal TV: allow fine detail
INV_RESTARTS     = 3
