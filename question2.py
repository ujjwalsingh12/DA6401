# Import necessary modules
from activation_func import ActivationFunctions
from test_func import TestingModel
from layers import layer
import numpy as np
import matplotlib.pyplot as plt
import copy
import wandb

class NeuralNetwork:
    def __init__(self, validation_images, validation_labels, loss_function):
        """
        Initialize the Neural Network with validation data and loss function.
        
        Args:
            validation_images: Validation input data
            validation_labels: Validation target labels
            loss_function: Type of loss function to use ('cross_entropy' or 'mean_squared_error')
        """
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.loss_function = loss_function
    
    def forward_pass(self, network_layers, input_data):
        """
        Perform forward propagation through the network.
        
        Args:
            network_layers: List of layer objects
            input_data: Input features
            
        Returns:
            Updated network_layers with computed activations
        """
        activation_functions = ActivationFunctions()
        
        # Process all layers except the output layer
        for layer_idx in range(len(network_layers) - 1):
            if layer_idx == 0:
                # First layer uses the input data
                network_layers[layer_idx].preactivation = network_layers[layer_idx].bias + np.dot(network_layers[layer_idx].weights, input_data)
                
                # Apply activation function
                if network_layers[layer_idx].activationFunction == 'sigmoid':
                    network_layers[layer_idx].activation = activation_functions.sigmoid(network_layers[layer_idx].preactivation)
                elif network_layers[layer_idx].activationFunction == 'tanh':
                    network_layers[layer_idx].activation = activation_functions.tanh(network_layers[layer_idx].preactivation)
                elif network_layers[layer_idx].activationFunction == 'relu':
                    network_layers[layer_idx].activation = activation_functions.relu(network_layers[layer_idx].preactivation)
            else:
                # Hidden layers use the previous layer's activation
                network_layers[layer_idx].preactivation = network_layers[layer_idx].bias + np.matmul(
                    network_layers[layer_idx].weights, network_layers[layer_idx-1].activation
                )
                
                # Apply activation function
                if network_layers[layer_idx].activationFunction == 'sigmoid':
                    network_layers[layer_idx].activation = activation_functions.sigmoid(network_layers[layer_idx].preactivation)
                elif network_layers[layer_idx].activationFunction == 'tanh':
                    network_layers[layer_idx].activation = activation_functions.tanh(network_layers[layer_idx].preactivation)
                elif network_layers[layer_idx].activationFunction == 'relu':
                    network_layers[layer_idx].activation = activation_functions.relu(network_layers[layer_idx].preactivation)
        
        # Process the output layer (always uses softmax)
        output_idx = len(network_layers) - 1
        network_layers[output_idx].preactivation = network_layers[output_idx].bias + np.dot(
            network_layers[output_idx].weights, network_layers[output_idx-1].activation
        )
        network_layers[output_idx].activation = activation_functions.softmax(network_layers[output_idx].preactivation)
        
        return network_layers

    def backward_pass(self, network_layers, input_data, target_label, regularization_lambda=0):
        """
        Perform backward propagation to compute gradients.
        
        Args:
            network_layers: List of layer objects
            input_data: Input features
            target_label: Target class label
            regularization_lambda: L2 regularization coefficient
            
        Returns:
            Updated network_layers with computed gradients
        """
        activation_functions = ActivationFunctions()
        
        # Create one-hot encoded target vector
        one_hot_target = np.zeros(network_layers[len(network_layers)-1].numberOfNeuronPerLayer)
        one_hot_target[target_label] = 1
        
        # Compute gradient for output layer
        output_idx = len(network_layers) - 1
        if self.loss_function == 'mean_squared_error':
            # For MSE, gradient includes derivative of softmax
            output_activation = network_layers[output_idx].activation
            network_layers[output_idx].gradient_preactivation = (output_activation - one_hot_target) * output_activation * (1 - output_activation)
        else:
            # For cross-entropy, gradient is simpler (softmax derivative cancels out)
            network_layers[output_idx].gradient_preactivation = network_layers[output_idx].activation - one_hot_target
        
        # Backpropagate through the network
        for layer_idx in range(len(network_layers) - 1, -1, -1):
            if layer_idx == 0:
                # For first layer, use input data to compute weight gradients
                network_layers[layer_idx].gradient_weights = np.outer(
                    network_layers[layer_idx].gradient_preactivation, input_data
                ) + regularization_lambda * network_layers[layer_idx].weights
            else:
                # For other layers, use previous layer's activation
                network_layers[layer_idx].gradient_weights = np.outer(
                    network_layers[layer_idx].gradient_preactivation, network_layers[layer_idx-1].activation
                ) + regularization_lambda * network_layers[layer_idx].weights
            
            # Bias gradients are the same as preactivation gradients
            network_layers[layer_idx].gradient_bias = network_layers[layer_idx].gradient_preactivation
            
            # Compute gradients for previous layer (if not at input layer)
            if layer_idx > 0:
                # Gradient of previous layer's activation
                network_layers[layer_idx-1].gradient_activation = np.matmul(
                    network_layers[layer_idx].weights.T, network_layers[layer_idx].gradient_preactivation
                )
                
                # Apply derivative of activation function
                if network_layers[layer_idx].activationFunction == 'sigmoid':
                    network_layers[layer_idx-1].gradient_preactivation = network_layers[layer_idx-1].gradient_activation * (
                        activation_functions.sigmoidGrad(network_layers[layer_idx-1].preactivation)
                    )
                elif network_layers[layer_idx].activationFunction == 'tanh':
                    network_layers[layer_idx-1].gradient_preactivation = network_layers[layer_idx-1].gradient_activation * (
                        activation_functions.tanhGrad(network_layers[layer_idx-1].preactivation)
                    )
                elif network_layers[layer_idx].activationFunction == 'relu':
                    network_layers[layer_idx-1].gradient_preactivation = network_layers[layer_idx-1].gradient_activation * (
                        activation_functions.reluGrad(network_layers[layer_idx-1].preactivation)
                    )
        
        return network_layers

    def sgd_optimizer(self, network_layers, max_epochs, train_images, train_labels, gradient_accumulator, 
                      batch_size, learning_rate, regularization_lambda):
        """
        Stochastic Gradient Descent optimizer.
        
        Args:
            network_layers: List of layer objects
            max_epochs: Maximum number of training epochs
            train_images: Training input data
            train_labels: Training target labels
            gradient_accumulator: List of layer objects to accumulate gradients
            batch_size: Mini-batch size
            learning_rate: Learning rate for weight updates
            regularization_lambda: L2 regularization coefficient
            
        Returns:
            Updated network_layers after training
        """
        for epoch in range(max_epochs):
            # Initialize gradients for each layer
            for layer_idx in range(len(network_layers)):
                network_layers[layer_idx].initialize_gradient()
                gradient_accumulator[layer_idx].initialize_gradient()
            
            batch_counter = 1
            
            # Process each training example
            for features, label in zip(train_images, train_labels):
                # Forward and backward pass
                network_layers = self.forward_pass(network_layers, features)
                network_layers = self.backward_pass(network_layers, features, label, regularization_lambda)
                
                # Accumulate gradients
                for layer_idx in range(len(network_layers)):
                    gradient_accumulator[layer_idx].gradient_weights += network_layers[layer_idx].gradient_weights
                    gradient_accumulator[layer_idx].gradient_bias += network_layers[layer_idx].gradient_bias
                
                # Update weights after each batch
                if batch_counter % batch_size == 0:
                    for layer_idx in range(len(network_layers)):
                        network_layers[layer_idx].weights -= learning_rate * gradient_accumulator[layer_idx].gradient_weights
                        network_layers[layer_idx].bias -= learning_rate * gradient_accumulator[layer_idx].gradient_bias
                        gradient_accumulator[layer_idx].initialize_gradient()
                
                batch_counter += 1
            
            # Evaluate and log performance after each epoch
            self._evaluate_and_log(network_layers, epoch, train_images, train_labels)
        
        return network_layers

    def momentum_optimizer(self, network_layers, max_epochs, train_images, train_labels, gradient_accumulator, 
                         batch_size, learning_rate, regularization_lambda):
        """
        Momentum-based gradient descent optimizer.
        
        Args:
            network_layers: List of layer objects
            max_epochs: Maximum number of training epochs
            train_images: Training input data
            train_labels: Training target labels
            gradient_accumulator: List of layer objects to accumulate gradients
            batch_size: Mini-batch size
            learning_rate: Learning rate for weight updates
            regularization_lambda: L2 regularization coefficient
            
        Returns:
            Updated network_layers after training
        """
        # Momentum hyperparameter
        momentum_beta = 0.9
        
        # Initialize velocity terms
        velocity_terms = []
        for layer_idx in range(len(network_layers)):
            velocity_terms.append(copy.deepcopy(network_layers[layer_idx]))
        
        for epoch in range(max_epochs):
            # Initialize gradients for each layer
            for layer_idx in range(len(network_layers)):
                network_layers[layer_idx].initialize_gradient()
                velocity_terms[layer_idx].initialize_gradient()
                gradient_accumulator[layer_idx].initialize_gradient()
            
            batch_counter = 1
            
            # Process each training example
            for features, label in zip(train_images, train_labels):
                # Forward and backward pass
                network_layers = self.forward_pass(network_layers, features)
                network_layers = self.backward_pass(network_layers, features, label, regularization_lambda)
                
                # Accumulate gradients
                for layer_idx in range(len(network_layers)):
                    gradient_accumulator[layer_idx].gradient_weights += network_layers[layer_idx].gradient_weights
                    gradient_accumulator[layer_idx].gradient_bias += network_layers[layer_idx].gradient_bias
                
                # Update weights after each batch
                if batch_counter % batch_size == 0:
                    for layer_idx in range(len(network_layers)):
                        # Update with momentum
                        network_layers[layer_idx].weights -= (momentum_beta * velocity_terms[layer_idx].gradient_weights + 
                                                              learning_rate * gradient_accumulator[layer_idx].gradient_weights)
                        network_layers[layer_idx].bias -= (momentum_beta * velocity_terms[layer_idx].gradient_bias + 
                                                           learning_rate * gradient_accumulator[layer_idx].gradient_bias)
                        
                        # Update velocity terms
                        velocity_terms[layer_idx].gradient_weights = (momentum_beta * velocity_terms[layer_idx].gradient_weights + 
                                                                      learning_rate * gradient_accumulator[layer_idx].gradient_weights)
                        velocity_terms[layer_idx].gradient_bias = (momentum_beta * velocity_terms[layer_idx].gradient_bias + 
                                                                   learning_rate * gradient_accumulator[layer_idx].gradient_bias)
                        
                        # Reset gradient accumulator
                        gradient_accumulator[layer_idx].initialize_gradient()
                
                batch_counter += 1
            
            # Evaluate and log performance after each epoch
            self._evaluate_and_log(network_layers, epoch, train_images, train_labels)
        
        return network_layers

def nesterov_optimizer(self, network_layers, max_epochs, train_images, train_labels, gradient_accumulator, batch_size, learning_rate, regularization_lambda):
    """
    Nesterov Accelerated Gradient optimizer.
    
    Args:
        network_layers: List of layer objects
        max_epochs: Maximum number of training epochs
        train_images: Training input data
        train_labels: Training target labels
        gradient_accumulator: List of layer objects to accumulate gradients
        batch_size: Mini-batch size
        learning_rate: Learning rate for weight updates
        regularization_lambda: L2 regularization coefficient
        
    Returns:
        Updated network_layers after training
    """
    # Momentum hyperparameter
    momentum_beta = 0.9
    
    # Initialize velocity and lookahead terms
    velocity_terms = []
    lookahead_terms = []
    
    # Deep copy the layer objects for momentum updates
    for layer_idx in range(len(network_layers)):
        velocity_terms.append(copy.deepcopy(network_layers[layer_idx]))
        lookahead_terms.append(copy.deepcopy(network_layers[layer_idx]))
    
    for epoch in range(max_epochs):
        # Initialize gradients for the current epoch
        for layer_idx in range(len(network_layers)):
            network_layers[layer_idx].initialize_gradient()
            gradient_accumulator[layer_idx].initialize_gradient()
        
        batch_counter = 1
        
        # Process each training example
        for features, label in zip(train_images, train_labels):
            # Forward and backward propagation
            network_layers = self.forward_pass(network_layers, features)
            network_layers = self.backward_pass(network_layers, features, label, regularization_lambda)
            
            # Accumulate gradients
            for layer_idx in range(len(network_layers)):
                gradient_accumulator[layer_idx].gradient_weights += network_layers[layer_idx].gradient_weights
                gradient_accumulator[layer_idx].gradient_bias += network_layers[layer_idx].gradient_bias
            
            # Update weights after each batch
            if batch_counter % batch_size == 0:
                # First apply lookahead step (move in the direction of the previous momentum)
                for layer_idx in range(len(network_layers)):
                    network_layers[layer_idx].weights += lookahead_terms[layer_idx].gradient_weights
                    network_layers[layer_idx].bias += lookahead_terms[layer_idx].gradient_bias
                
                # Compute new momentum-based update
                for layer_idx in range(len(network_layers)):
                    # Compute new velocity term with momentum
                    lookahead_terms[layer_idx].gradient_weights = (momentum_beta * velocity_terms[layer_idx].gradient_weights + 
                                                                  learning_rate * gradient_accumulator[layer_idx].gradient_weights)
                    lookahead_terms[layer_idx].gradient_bias = (momentum_beta * velocity_terms[layer_idx].gradient_bias + 
                                                               learning_rate * gradient_accumulator[layer_idx].gradient_bias)
                    
                    # Apply updated gradients to weights and biases
                    network_layers[layer_idx].weights -= lookahead_terms[layer_idx].gradient_weights
                    network_layers[layer_idx].bias -= lookahead_terms[layer_idx].gradient_bias
                    
                    # Store current velocity for next iteration
                    velocity_terms[layer_idx].gradient_weights = lookahead_terms[layer_idx].gradient_weights
                    velocity_terms[layer_idx].gradient_bias = lookahead_terms[layer_idx].gradient_bias
                    
                    # Reset gradient accumulator
                    gradient_accumulator[layer_idx].initialize_gradient()
                
                # Undo the initial lookahead step (crucial for Nesterov's method)
                for layer_idx in range(len(network_layers)):
                    network_layers[layer_idx].weights -= lookahead_terms[layer_idx].gradient_weights
                    network_layers[layer_idx].bias -= lookahead_terms[layer_idx].gradient_bias
            
            batch_counter += 1
        
        # Evaluate and log metrics after each epoch
        self._evaluate_and_log(network_layers, epoch, train_images, train_labels)
    
    return network_layers

def _evaluate_and_log(self, network_layers, epoch, train_images, train_labels):
    """
    Evaluate model performance and log metrics after each epoch.
    
    Args:
        network_layers: List of layer objects (trained model)
        epoch: Current epoch number
        train_images: Training input data
        train_labels: Training target labels
    """
    # Create a testing model instance
    test = TestingModel()
    
    # Create a deep copy of the network for evaluation
    eval_network = copy.deepcopy(network_layers)
    
    if self.loss_function == 'cross_entropy':
        # Calculate training loss and accuracy
        train_loss = test.crossEntropyLoss(eval_network, train_images, train_labels)
        train_predictions = test.CalculateTest(eval_network, train_images)
        train_accuracy = test.zeroOneModel(train_predictions, train_labels)
        
        # Calculate validation loss and accuracy
        val_loss = test.crossEntropyLoss(eval_network, self.validation_images, self.validation_labels)
        val_predictions = test.CalculateTest(eval_network, self.validation_images)
        val_accuracy = test.zeroOneModel(val_predictions, self.validation_labels)
        
        # Print performance metrics
        print(f"Epoch number: {epoch+1}")
        print(f"Cross Entropy train Loss: {train_loss}, Accuracy: {train_accuracy}")
        print(f"Cross Entropy val Loss: {val_loss}, Accuracy: {val_accuracy}")
    else:
        # Calculate training loss and accuracy with squared error
        train_loss = test.squareError(eval_network, train_images, train_labels)
        train_predictions = test.CalculateTest(eval_network, train_images)
        train_accuracy = test.zeroOneModel(train_predictions, train_labels)
        
        # Calculate validation loss and accuracy
        val_loss = test.squareError(eval_network, self.validation_images, self.validation_labels)
        val_predictions = test.CalculateTest(eval_network, self.validation_images)
        val_accuracy = test.zeroOneModel(val_predictions, self.validation_labels)
        
        # Print performance metrics
        print(f"Epoch number: {epoch+1}")
        print(f"Square Error train Loss: {train_loss}, Accuracy: {train_accuracy}")
        print(f"Square Error val Loss: {val_loss}, Accuracy: {val_accuracy}")
    
    # Log metrics to wandb
    wandb.log({
        "Training_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "validation_loss": val_loss,
        "Training_loss": train_loss,
        "epoch": epoch+1
    })
    
def RmsProp(self, network_layers, epochs_max, training_data, training_labels, network_gradients, mini_batch_size, learning_rate, reg_lambda):
    # Set constants for RMSProp algorithm
    decay_rate, epsilon = 0.5, 1e-8
    gradient_cache = []
    for i in range(len(network_layers)):      # Initialize history for squared gradients
      gradient_cache.append(copy.deepcopy(network_layers[i]))
      gradient_cache[i].initialize_gradient()
    for epoch in range(epochs_max):            # Iterate through epochs
      for j in range(len(network_layers)):       # Initialize gradients for current epoch
        network_layers[j].initialize_gradient()
        network_gradients[j].initialize_gradient()
      batch_counter = 1
      for input_data, target in zip(training_data, training_labels):         # Iterate through training data batches
        network_layers = self.forwardPropagation(network_layers, input_data)
        network_layers = self.backPropagation(network_layers, input_data, target, reg_lambda)
        for i in range(len(network_layers)):                 # Accumulate gradients
          network_gradients[i].gradw = network_gradients[i].gradw + network_layers[i].gradw
          network_gradients[i].gradb = network_gradients[i].gradb + network_layers[i].gradb
        if batch_counter % mini_batch_size == 0:                             # Update weights and biases after every batch
          for i in range(len(network_layers)):
            gradient_cache[i].gradw = decay_rate * gradient_cache[i].gradw + (1 - decay_rate) * np.square(network_gradients[i].gradw)
            gradient_cache[i].gradb = decay_rate * gradient_cache[i].gradb + (1 - decay_rate) * np.square(network_gradients[i].gradb)
            network_layers[i].w = network_layers[i].w - ((learning_rate * network_gradients[i].gradw) / (np.sqrt((gradient_cache[i].gradw + epsilon))))
            network_layers[i].b = network_layers[i].b - ((learning_rate * network_gradients[i].gradb) / (np.sqrt((gradient_cache[i].gradb + epsilon))))
            network_gradients[i].initialize_gradient()
        batch_counter = batch_counter + 1
      if self.loss == 'cross_entropy':                   # Compute and log losses and accuracies after each epoch if loss is cross entropy
        evaluator = TestingModel()
        current_layers = copy.deepcopy(network_layers)
        train_loss = evaluator.crossEntropyLoss(current_layers, training_data, training_labels)
        val_loss = evaluator.crossEntropyLoss(current_layers, self.train_image_val, self.train_class_val)
        train_predictions = evaluator.CalculateTest(current_layers, training_data)
        train_accuracy = evaluator.zeroOneModel(train_predictions, training_labels)
        val_predictions = evaluator.CalculateTest(current_layers, self.train_image_val)
        val_accuracy = evaluator.zeroOneModel(val_predictions, self.train_class_val)
        print("epoch number is ", epoch + 1)
        print("Cross Entropy train Loss is :", train_loss, "Accuracy is :", train_accuracy)
        print("Cross Entropy val_Loss is :", val_loss, "Accuracy is :", val_accuracy)
        wandb.log({"Training_accuracy": train_accuracy, "val_accuracy": val_accuracy, 
                   "validation_loss": val_loss, "Training_loss": train_loss, "epoch": epoch + 1})
      else:                                            #Compute and log losses and accuracies after each epoch if loss is square loss
        evaluator = TestingModel()
        current_layers = copy.deepcopy(network_layers)
        train_loss = evaluator.squareError(current_layers, training_data, training_labels)
        val_loss = evaluator.squareError(current_layers, self.train_image_val, self.train_class_val)
        train_predictions = evaluator.CalculateTest(current_layers, training_data)
        train_accuracy = evaluator.zeroOneModel(train_predictions, training_labels)
        val_predictions = evaluator.CalculateTest(current_layers, self.train_image_val)
        val_accuracy = evaluator.zeroOneModel(val_predictions, self.train_class_val)
        print("epoch number is ", epoch + 1)
        print("squareError train Loss is :", train_loss, "Accuracy is :", train_accuracy)
        print("squareError val_Loss is :", val_loss, "Accuracy is :", val_accuracy)
        wandb.log({"Training_accuracy": train_accuracy, "val_accuracy": val_accuracy, 
                   "validation_loss": val_loss, "Training_loss": train_loss, "epoch": epoch + 1})
    return network_layers

def Adam_optimizer(self, layers, max_epochs, train_data, train_labels, layer_grads, batch_sz, learning_rate, reg_coeff):
    beta, epsilon = 0.5, 1e-10
    beta1, beta2 = 0.9, 0.999
    momentum_cache, rms_cache = [], []
    
    for layer in layers:
        momentum_cache.append(copy.deepcopy(layer))
        rms_cache.append(copy.deepcopy(layer))
        momentum_cache[-1].initialize_gradient()
        rms_cache[-1].initialize_gradient()
    
    for epoch in range(max_epochs):
        for layer, grad in zip(layers, layer_grads):
            layer.initialize_gradient()
            grad.initialize_gradient()
        
        count = 1
        for data, label in zip(train_data, train_labels):
            layers = self.forwardPropagation(layers, data)
            layers = self.backPropagation(layers, data, label, reg_coeff)
            
            for i in range(len(layers)):
                layer_grads[i].gradw += layers[i].gradw
                layer_grads[i].gradb += layers[i].gradb
            
            if count % batch_sz == 1:
                for i in range(len(layers)):
                    momentum_cache[i].gradw = beta1 * momentum_cache[i].gradw + (1 - beta1) * layer_grads[i].gradw
                    momentum_cache[i].gradb = beta1 * momentum_cache[i].gradb + (1 - beta1) * layer_grads[i].gradb
                    rms_cache[i].gradw = beta2 * rms_cache[i].gradw + (1 - beta2) * np.square(layer_grads[i].gradw)
                    rms_cache[i].gradb = beta2 * rms_cache[i].gradb + (1 - beta2) * np.square(layer_grads[i].gradb)
                    
                    mw_hat = momentum_cache[i].gradw / (1 - np.power(beta1, epoch + 1))
                    mb_hat = momentum_cache[i].gradb / (1 - np.power(beta1, epoch + 1))
                    vw_hat = rms_cache[i].gradw / (1 - np.power(beta2, epoch + 1))
                    vb_hat = rms_cache[i].gradb / (1 - np.power(beta2, epoch + 1))
                    
                    layers[i].w -= (learning_rate * mw_hat) / (np.sqrt(vw_hat + epsilon))
                    layers[i].b -= (learning_rate * mb_hat) / (np.sqrt(vb_hat + epsilon))
                    
                    layer_grads[i].initialize_gradient()
            
            count += 1
        
        if self.loss == 'cross_entropy':
            test_model = TestingModel()
            cloned_layers = copy.deepcopy(layers)
            train_loss = test_model.crossEntropyLoss(cloned_layers, train_data, train_labels)
            val_loss = test_model.crossEntropyLoss(cloned_layers, self.train_data_val, self.train_labels_val)
            train_accuracy = test_model.zeroOneModel(test_model.CalculateTest(cloned_layers, train_data), train_labels)
            val_accuracy = test_model.zeroOneModel(test_model.CalculateTest(cloned_layers, self.train_data_val), self.train_labels_val)
            
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Accuracy = {train_accuracy}")
            print(f"Validation Loss = {val_loss}, Accuracy = {val_accuracy}")
            wandb.log({"Train Accuracy": train_accuracy, "Validation Accuracy": val_accuracy, "Validation Loss": val_loss, "Train Loss": train_loss, "Epoch": epoch + 1})
        else:
            test_model = TestingModel()
            cloned_layers = copy.deepcopy(layers)
            train_loss = test_model.squareError(cloned_layers, train_data, train_labels)
            val_loss = test_model.squareError(cloned_layers, self.train_data_val, self.train_labels_val)
            train_accuracy = test_model.zeroOneModel(test_model.CalculateTest(cloned_layers, train_data), train_labels)
            val_accuracy = test_model.zeroOneModel(test_model.CalculateTest(cloned_layers, self.train_data_val), self.train_labels_val)
            
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Accuracy = {train_accuracy}")
            print(f"Validation Loss = {val_loss}, Accuracy = {val_accuracy}")
            wandb.log({"Train Accuracy": train_accuracy, "Validation Accuracy": val_accuracy, "Validation Loss": val_loss, "Train Loss": train_loss, "Epoch": epoch + 1})
    
    return layers