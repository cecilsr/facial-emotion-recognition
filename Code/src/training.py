"""Training and evaluation utilities for emotion classifiers."""

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(device, model, train_loader, val_loader, lr, beta1=0.9, beta2=0.999, num_epochs=2, verbose=False):
    """Trains The passed model with given hyperparameters and return validation accuracy.

    Parameters
    ----------
    device : torch.device
        Execution device.
    model : torch.nn.Module
        Model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    lr : float
        Learning rate.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    num_epochs : int
        Number of epochs.
    verbose : bool
        Print training loss if True.

    Returns
    -------
    float
        Validation accuracy in percent.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, sample in enumerate(train_loader):
            data, target = sample["image"], sample["emolabel"]
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose:
            avg_loss = running_loss / len(train_loader)
            print(f"[lr={lr}, beta1={beta1}, beta2={beta2}] "
                  f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    acc = evaluate(device, model, val_loader)
    return acc


def evaluate(device, model, loader):
    """
    Compute classification accuracy.

    Parameters
    ----------
    device : torch.device
        Execution device.
    model : torch.nn.Module
        Model to evaluate.
    loader : DataLoader
        Evaluation data loader.

    Returns
    -------
    float
        Accuracy in percent.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in loader:
            data, target = sample["image"], sample["emolabel"]
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy