import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
import wandb
import math
import torch.nn.functional as F
import getdata

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, val_loader, train_loader, epoch):
    model.eval()
    val_correct = 0
    val_total = 0
    train_correct = 0
    train_total = 0
    val_loss = 0
    train_loss = 0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_loss += criterion(outputs, labels).item()

    wandb.log({
        'epoch': epoch,
        'val_accuracy': 100 * val_correct / val_total,
        'val_loss': val_loss,
        'train_accuracy': 100 * train_correct / train_total,
        'train_loss': train_loss,
    })
    print(f"Epoch {epoch}: Val Acc = {val_correct}/{val_total} ({100 * val_correct / val_total:.2f}%)")

class CustomCNN(nn.Module):
    def __init__(self, config):
        super(CustomCNN, self).__init__()
        self.activation_type = config.activation
        self.base_filters = config.filters
        self.filter_strategy = config.filter_organisation
        self.kernel = config.kernel_size
        self.dropout_prob = config.dropout
        self.hidden_nodes = config.hl_nodes
        self.batchnorm = config.batch_normalisation

        pad = self.kernel // 2
        self.pool = nn.MaxPool2d(2, 2)
        self.f1 = self.get_next_filter()
        self.f2 = self.get_next_filter()
        self.f3 = self.get_next_filter()
        self.f4 = self.get_next_filter()
        self.f5 = self.get_next_filter()

        self.conv1 = nn.Conv2d(3, self.f1, self.kernel, 1, pad)
        self.conv2 = nn.Conv2d(self.f1, self.f2, self.kernel, 1, pad)
        self.conv3 = nn.Conv2d(self.f2, self.f3, self.kernel, 1, pad)
        self.conv4 = nn.Conv2d(self.f3, self.f4, self.kernel, 1, pad)
        self.conv5 = nn.Conv2d(self.f4, self.f5, self.kernel, 1, pad)

        self.bn1 = nn.BatchNorm2d(self.f1)
        self.bn2 = nn.BatchNorm2d(self.f2)
        self.bn3 = nn.BatchNorm2d(self.f3)
        self.bn4 = nn.BatchNorm2d(self.f4)
        self.bn5 = nn.BatchNorm2d(self.f5)
        self.bn_hidden = nn.BatchNorm1d(self.hidden_nodes)

        self.fc1 = nn.Linear(self.base_filters * 8 * 8, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, 10)

    def get_next_filter(self):
        if self.filter_strategy == 'half':
            self.base_filters = math.ceil(self.base_filters / 2)
        elif self.filter_strategy == 'double':
            self.base_filters *= 2
        return self.base_filters

    def activation(self, x):
        if self.activation_type == 'ReLU':
            return F.relu(x)
        elif self.activation_type == 'GeLU':
            return F.gelu(x)
        elif self.activation_type == 'SiLU':
            return F.silu(x)
        else:
            return x * torch.tanh(F.softplus(x))  # Mish

    def forward(self, x):
        x = x.to(device)

        x = self.activation(self.bn1(self.conv1(x)) if self.batchnorm == 'yes' else self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.bn2(self.conv2(x)) if self.batchnorm == 'yes' else self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.bn3(self.conv3(x)) if self.batchnorm == 'yes' else self.conv3(x))
        x = self.pool(x)
        x = self.activation(self.bn4(self.conv4(x)) if self.batchnorm == 'yes' else self.conv4(x))
        x = self.pool(x)
        x = self.activation(self.bn5(self.conv5(x)) if self.batchnorm == 'yes' else self.conv5(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = nn.Dropout(self.dropout_prob)(x)
        x = self.activation(self.bn_hidden(self.fc1(x)) if self.batchnorm == 'yes' else self.fc1(x))
        x = self.fc2(x)

        return x

def run_training():
    wandb.init(project="model_eval", name="test")
    config = wandb.config

    train_set, val_set, test_set = get_data(config.data_augmentation)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    model = CustomCNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 500 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

        evaluate_model(model, val_loader, train_loader, epoch + 1)

    wandb.finish()
    
# Best model parameters
def main ():
    sweep_config = {
    'method': 'bayes',
    'name' : 'testing',
    'metric': {
      'name': 'validation_accuracy',  # <- updated to match wandb.log in evaluate()
      'goal': 'maximize'
    },
    'parameters': {
      'filters':{
          'values':[16,32,64]
      },
      'activation':{
          'values':['GeLU','ReLU','SiLU']

      },
      'filter_organisation':{
          'values':['same']
      },
      'data_augmentation':{
          'values':['yes']
      },
      'batch_normalisation':{
          'values':['no']
      },
      'dropout':{
          'values':[0.2,0.1,0.05]
      },
      'batch_size':{
          'values':[32,64,128]
      },
      'kernel_size':{
         'values':[3]
      },
      'learning_rate':{
         'values':[0.001]
      },
      'momentum':{
         'values':[0.9]
      },
      'epochs':{
         'values':[10,15]
      },
      'hl_nodes':{
         'values':[256]
      }
    }
}

    sweep_id = wandb.sweep(sweep=sweep_config, project='DA6401_A2', entity='cs23m071-indian-institute-of-technology-madras')

    # Execute the sweep
    wandb.agent(sweep_id, function=run_training, count=50)

if __name__ == "__main__":
    wandb.login(key="1b74d87eef0c8dff900595f1526e95e162049f6a")
    main()
