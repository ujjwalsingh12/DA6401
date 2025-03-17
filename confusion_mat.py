from activation_func import ActivationFunctions
from question2 import NeuralNetwork
from test_func import TestingModel
from layers import layer
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import wandb

wandb.login()

def generate_confusion_matrix(predictions, actual_labels):
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    wandb.log({"Confusion Matrix": wandb.sklearn.plot_confusion_matrix(actual_labels, predictions, class_labels)})

def main():
    total_layers = 3
    neurons_per_hidden_layer = 128
    output_neurons = 10
    activation_fn = 'tanh'
    weight_init = 'xavier'
    learning_rate = 0.01
    reg_coefficient = 0
    batch_size = 64
    optimizer_type = 'adam'
    total_epochs = 15
    loss_function = "cross_entropy"
    dataset_choice = 'fashion_mnist'

    if dataset_choice == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        reshaped_train = train_images.reshape(train_images.shape[0], -1)
        validation_images = reshaped_train[int(0.9 * reshaped_train.shape[0]):]
        validation_labels = train_labels[int(0.9 * reshaped_train.shape[0]):]
        train_images = reshaped_train[:int(0.9 * reshaped_train.shape[0])]
        train_labels = train_labels[:int(0.9 * reshaped_train.shape[0])]
        train_images /= 256
        validation_images /= 256
        reshaped_test = test_images.reshape(test_images.shape[0], -1)
        reshaped_test /= 256
    else:
        print('Using MNIST dataset')
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        reshaped_train = train_images.reshape(train_images.shape[0], -1)
        validation_images = reshaped_train[int(0.9 * reshaped_train.shape[0]):]
        validation_labels = train_labels[int(0.9 * reshaped_train.shape[0]):]
        train_images = reshaped_train[:int(0.9 * reshaped_train.shape[0])]
        train_labels = train_labels[:int(0.9 * reshaped_train.shape[0])]
        train_images /= 256
        validation_images /= 256
        reshaped_test = test_images.reshape(test_images.shape[0], -1)
        reshaped_test /= 256

    previous_layer_neurons = train_images.shape[1]
    network_layers = []
    gradient_layers = []
    
    for layer_index in range(total_layers):
        if layer_index == total_layers - 1:
            layer_obj = layer(output_neurons, previous_layer_neurons, weight_init, activation_fn)
        else:
            layer_obj = layer(neurons_per_hidden_layer, previous_layer_neurons, weight_init, activation_fn)
            previous_layer_neurons = neurons_per_hidden_layer
        
        network_layers.append(layer_obj)
        gradient_layers.append(copy.deepcopy(layer_obj))
    
    model_trainer = NeuralNetwork(validation_images, validation_labels, loss_function)
    wandb.init(project='DA6401-Assignment', magic=False, reinit=True)
    network_layers = model_trainer.Adam_gradient_descent(network_layers, total_epochs, train_images, train_labels, gradient_layers, batch_size, learning_rate, reg_coefficient)
    
    tester = TestingModel()
    predictions = tester.CalculateTest(network_layers, reshaped_test)
    test_accuracy = tester.zeroOneModel(predictions, test_labels)
    test_loss = tester.crossEntropyLoss(network_layers, reshaped_test, test_labels)
    
    print("Test Accuracy:", test_accuracy)
    print("Test Loss:", test_loss)
    generate_confusion_matrix(predictions, list(test_labels))

if __name__ == "__main__":
    main()