# FedAvg - Federated Learning Aggregation Method

Implementation of the FedAvg (Federated Averaging algorithm) from the paper
"Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al.), with dataset Fashion MNIST.

## Setup

### 1. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### 2. Run

```bash
cd src
python main.py
```

At the beginning of each run you are asked to log metrics on WandB. To skip it, insert "3".
