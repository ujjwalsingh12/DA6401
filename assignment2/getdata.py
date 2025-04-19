import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

# Give path to train folder
train_data_path = "./inaturalist_12K/train"
# Give path to val folder
val_data_path = "./inaturalist_12K/val"

# Transform the data
data_transforms = {
    'trainaugmentation': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'train':transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}
# Load the data and pass to te function
def get_data (augmentation):
    train_dataset = ImageFolder(root=train_data_path, transform=data_transforms['trainaugmentation'] if augmentation == 'yes' else data_transforms['train'])
    validation_size = int(0.2 * len(train_dataset))
    training_size = len(train_dataset) - validation_size
    test_dataset = ImageFolder(root=val_data_path, transform=data_transforms['val'])
    training_data, validation_data = random_split(train_dataset, [training_size, validation_size])
    return training_data,validation_data,test_dataset
