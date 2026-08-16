import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


#ClientUpdate function
def client_update(client_model, dataset, indices, epochs, batch_size, lr):
    client_model.train()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    #creazione DataLoader con solo i dati per il singolo client
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0

    #addestramento presso il singolo client per E epoche
    for epoch in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            output = client_model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    return client_model.state_dict(), len(indices), avg_loss
