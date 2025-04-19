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
import getdata from getdata

# For GPU Run
device =  'cuda' if torch.cuda.is_available() else 'cpu'
# Function for evaluating accuracy and runnning tests
def evaluate(model,val_data,train_data,epoch):
   model.eval()
   vcorrect=0
   vtotal = 0
   tcorrect = 0
   ttotal =0
   vloss=0
   tloss=0

   with torch.no_grad() :
    for data in val_data :
         images,labels = data 
         images = images.to(device)
         labels=labels.to(device)
         pred = model(images)
         _, predicted = torch.max(pred.data, 1)
         vtotal += labels.size(0)
         vcorrect += (predicted == labels).sum().item()
         criterion = nn.CrossEntropyLoss()
         vloss = criterion(pred, labels) 
    for data in train_data:
         images,labels = data 
         images = images.to(device)
         labels=labels.to(device)
         pred = model(images)
         _, predicted = torch.max(pred.data, 1)
         ttotal += labels.size(0)
         tcorrect += (predicted == labels).sum().item()
         criterion = nn.CrossEntropyLoss()
         tloss = criterion(pred, labels) 
    wandb.log({
            'epoch':epoch,
            'validation_accuracy':vcorrect*100/vtotal ,
            'validation_loss':vloss,
            'training_accuracy':tcorrect*100/ttotal,
            'training_loss':tloss,
         })
    print(vcorrect,vtotal,epoch)

      

# class for CNN model
class CNNmodel (nn.Module):
  def __init__ (self,config) :
    super(CNNmodel,self).__init__()
    self.activation = config.activation
    self.filters=config.filters
    self.filter_augmentation =config.filter_organisation
    self.kernel_size=config.kernel_size
    self.dropout = config.dropout
    self.hl_nodes = config.hl_nodes
    mypadding = self.kernel_size//2
    self.batch_normalization=config.batch_normalisation
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.out1=self.getfilters()
    self.out2=self.getfilters()
    self.out3=self.getfilters()
    self.out4=self.getfilters()
    self.out5=self.getfilters()

    self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.out1,kernel_size=self.kernel_size,stride=1,padding=mypadding)

    self.conv2 = nn.Conv2d(in_channels=self.out1,out_channels=self.out2,kernel_size=self.kernel_size,stride=1,padding=mypadding)

    self.conv3 = nn.Conv2d(in_channels=self.out2,out_channels=self.out3,kernel_size=self.kernel_size,stride=1,padding=mypadding)

    self.conv4 = nn.Conv2d(in_channels=self.out3,out_channels=self.out4,kernel_size=self.kernel_size,stride=1,padding=mypadding)

    self.conv5 = nn.Conv2d(in_channels=self.out4,out_channels=self.out5,kernel_size=self.kernel_size,stride=1,padding=mypadding)
    self.bnhl = nn.BatchNorm1d(self.hl_nodes)
    self.bn1 = nn.BatchNorm2d(self.out1)
    self.bn2 = nn.BatchNorm2d(self.out2)
    self.bn3 = nn.BatchNorm2d(self.out3)
    self.bn4 = nn.BatchNorm2d(self.out4)
    self.bn5 = nn.BatchNorm2d(self.out5)
    self.fc1=nn.Linear(in_features=self.filters*(8*8),out_features=self.hl_nodes)
    self.fc2=nn.Linear(in_features=self.hl_nodes,out_features=10)

  def activation_op (self,x):
    # if (self.batch_normalization=='yes'):
    #   bn = nn.BatchNorm2d(self.filters)
    #   x=bn(x)
    if self.activation == 'ReLU' :
        return F.relu(x)
    elif self.activation == 'GeLU' :
       return F.gelu(x)
    elif self.activation == 'SiLU':
      return  F.silu(x)
    else:
        return x * torch.tanh(F.softplus(x))
 
  def getfilters (self):
     if self.filter_augmentation == 'half':
        self.filters = math.ceil(self.filters/2)
     elif self.filter_augmentation == 'double':
        self.filters=2*self.filters
     return self.filters
        
     
     
# Defining the CNN model
  def forward (self,x):
    x=x.to(device)
    x=self.conv1(x)
    if (self.batch_normalization=='yes'):
       x=self.bn1(x)
    x=self.activation_op(x)
    x=self.pool(x)
    x=self.conv2(x)
    if (self.batch_normalization=='yes'):
       x=self.bn2(x)
    x=self.activation_op(x)
    x=self.pool(x)
    x=self.conv3(x)
    if (self.batch_normalization=='yes'):
       x=self.bn3(x)
    x=self.activation_op(x)
    x=self.pool(x)
    x=self.conv4(x)
    if (self.batch_normalization=='yes'):
       x=self.bn4(x)
    x=self.activation_op(x)
    x=self.pool(x)
    x=self.conv5(x)
    if (self.batch_normalization=='yes'):
       x=self.bn5(x)
    x=self.activation_op(x)
    x=self.pool(x)


    # x = x.view(-1, self.filters * 8 * 8)
    x = torch.flatten(x, start_dim=1)
    dropoutcode = nn.Dropout(p=self.dropout)
    x=dropoutcode(x)
    x=self.fc1(x)
    if (self.batch_normalization=='yes'):
       x=self.bnhl(x)
    x=self.activation_op(x)
    
    if (self.batch_normalization=='yes'):
        x=self.bnhl(x)
    
    
    x = self.fc2(x)
    return x

def train_wandb():

    # Initialize wandb
    
    
    wandb.init( )
    # Access the hyperparameters defined in the sweep
    config = wandb.config

    training_data,validation_data,test_data=get_data(config.data_augmentation)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size=config.batch_size, shuffle=False)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=config.batch_size, shuffle=False)
    
    model = CNNmodel(config)
    model=model.to(device)
    criterion = nn.CrossEntropyLoss()
    # using adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    for j in range (0,config.epochs) :
       current_loss = 0.0
       model.train()
       for i, data in enumerate(train_loader, 0):
          inp, labels = data
          inp = inp.to(device)
          labels=labels.to(device)
          optimizer.zero_grad()
          op = model(inp)
          loss = criterion(op, labels) 
          loss.backward() 
          optimizer.step() 
          if(i%500==0):
             print(i) 

          current_loss += loss.item()
       evaluate(model,val_loader,train_loader,j+1)

    # Train the model using the specified optimizer and hyperparameters

    wandb.finish()

    # Best model parameters
def main ():
    sweep_config = {
    'method': 'bayes',
    'name' : 'question 1',
    'metric': {
      'name': 'test_accuracy',
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

# # Execute the sweep
    wandb.agent(sweep_id, function=train_wandb, count=5)


if __name__ == "__main__":
    wandb.login(key="1b74d87eef0c8dff900595f1526e95e162049f6a")
    main()
