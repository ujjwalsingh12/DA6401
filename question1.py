
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import wandb
wandb.login(key="1b74d87eef0c8dff900595f1526e95e162049f6a")

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['Tshirt', 'Pants', 'Sweater', 'Gown', 'Jacket',  
               'Slipper', 'Shirt', 'Shoes', 'Handbag', 'Boots']  

# Create a figure
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

# Track found classes
found_classes = {}

# Iterate over dataset and select one sample per class
for i, (image, label) in enumerate(zip(train_images, train_labels)):
    if label not in found_classes:
        found_classes[label] = image
        if len(found_classes) == 10:  # Stop when all 10 classes are found
            break

# Plot the images
for i, (label, image) in enumerate(found_classes.items()):
    ax = axes[i // 5, i % 5]
    ax.imshow(image, cmap='gray')
    ax.set_title(class_names[label])
    ax.axis('off')

plt.tight_layout()
plt.show()

wandb.init(project="Deep Learning Course DA6401", name="Question 1")
wandb_images = []

for label, image in found_classes.items():
    wandb_images.append(wandb.Image(image, caption=class_names[label]))

# Log images to WandB
wandb.log({"Fashion-MNIST Samples": wandb_images})

# Finish WandB logging

