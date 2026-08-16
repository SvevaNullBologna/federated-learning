#!/usr/bin/env python
# coding: utf-8

#utilizzo dataset Fashion MNIST

from simulation import run_federated_averaging


#esecuzione simulazione
if __name__ == "__main__":
    print("\n Esperimento 1: Partizionamento IID\n")
    run_federated_averaging(iid=True, num_rounds=50, local_epochs=5, batch_size=10, fraction=0.1)

    print("\n\n Esperimento 2: Partizionamento Non-IID\n")
    run_federated_averaging(iid=False, num_rounds=50, local_epochs=5, batch_size=10, fraction=0.1)
