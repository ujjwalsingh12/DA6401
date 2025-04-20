import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
import wandb
from create_data import get_data

# Use GPU if available
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


# Evaluation function
def evaluate(model, val_loader, train_loader, epoch_num):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    train_correct = 0
    train_total = 0
    train_loss = 0.0

    with torch.no_grad():
        # Validation metrics
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

        # Training metrics
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_loss += criterion(outputs, labels).item()

    # Log metrics to wandb
    wandb.log({
        'epoch': epoch_num,
        'validation_accuracy': val_correct * 100 / val_total,
        'validation_loss': val_loss / len(val_loader),
        'training_accuracy': train_correct * 100 / train_total,
        'training_loss': train_loss / len(train_loader),
    })

    print(f"Epoch {epoch_num}: Validation Accuracy = {val_correct}/{val_total}")


# Training function
def train_wandb():
    wandb.login()
    wandb.init()
    config = wandb.config

    train_dataset, val_dataset, test_dataset = get_data(config.data_augmentation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load pretrained ResNet50
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        evaluate(model, val_loader, train_loader, epoch + 1)

    wandb.finish()


# Sweep configuration
def main():
    sweep_config = {
        'method': 'bayes',
        'name': 'resnet50_finetune_sweep',
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
            'batch_size': {'values': [16, 32]},
            'epochs': {'values': [5, 10]},
            'data_augmentation': {'values': ['yes']},
            'freeze_layers': {'values': ['all_but_last', 'last_k', 'none']}  # Optional use
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project='DA6401_A2', entity='cs23m071-indian-institute-of-technology-madras')
    wandb.agent(sweep_id, function=train_wandb, count=200)


if __name__ == "__main__":
    wandb.login(key="1b74d87eef0c8dff900595f1526e95e162049f6a")
    main()
