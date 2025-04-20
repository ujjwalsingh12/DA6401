import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

# Paths to dataset
train_directory = "./inaturalist_12K/train"
# Give path to val folder
val_directory= "./inaturalist_12K/val"

# Define transformation pipelines
transform_pipelines = {
    'augmented_train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'standard_val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'standard_train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}

# Function to load and split data
def get_data(apply_augmentation):
    full_train_dataset = ImageFolder(
        root=train_directory,
        transform=transform_pipelines['augmented_train'] if apply_augmentation == 'yes' else transform_pipelines['standard_train']
    )

    validation_split = int(0.2 * len(full_train_dataset))
    training_split = len(full_train_dataset) - validation_split

    training_dataset, validation_dataset = random_split(full_train_dataset, [training_split, validation_split])

    test_dataset = ImageFolder(root=val_directory, transform=transform_pipelines['standard_val'])

    return training_dataset, validation_dataset, test_dataset
