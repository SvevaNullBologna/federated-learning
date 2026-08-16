import torch
import numpy as np
import random


#partizionamento dei dati IID
def dataset_partition_IID(dataset, num_clients):
    indices = np.random.permutation(len(dataset))
    chunks = np.array_split(indices, num_clients)
    return {i: chunks[i].tolist() for i in range(num_clients)}


#partizionamento dei dati NON_IID
def dataset_partition_not_IID(dataset, num_clients, shards_per_client=2):
   labels = np.array([dataset.targets[i].item() if torch.is_tensor(dataset.targets[i]) else dataset.targets[i] for i in range(len(dataset))])
   sorted_indices = np.argsort(labels)

   num_shards = num_clients * shards_per_client
   shard_size = len(dataset) // num_shards
   shards = [sorted_indices[i*shard_size:(i+1)*shard_size].tolist() for i in range(num_shards)]

   random.shuffle(shards)

   client_data = {}
   for i in range(num_clients):
      client_data[i] = shards[i*shards_per_client] + shards[i*shards_per_client +1]

   return client_data
