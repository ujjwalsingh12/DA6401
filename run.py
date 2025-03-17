# Importing necessary libraries and modules
from activation_func import ActivationFunctions
from question2 import NeuralNetwork
from test_func import TestingModel
from layers import layer
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import wandb
import argparse
# Logging into Weights & Biases
wandb.login()
#760091de6b192857b226ee4bdecf4e7f93175087

def confusion_matrix_create(pred,true_p):   # Function to create confusion matrix
  #print('hi')
  label_class=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
  wandb.log({ "Confusion Matrix" : wandb.sklearn.plot_confusion_matrix(true_p,pred,label_class)})
# Main function
def main(config=None):
  if 2==2:
    #with wandb.init(config=config, ):
    #config=wandb.config
    run_name = 'bs-'+str(args.batch_size)+'-lr-'+ str(args.learning_rate)+'-ep-'+str(args.epochs)+ '-op-'+str(args.optimizer)+ '-nhl-'+str(args.num_layers)+'-shl-'+str(args.hidden_size)+ '-act-'+str(args.activation)+'-wd-'+str(args.weight_decay)+'-wi-'+str(args.weight_init)+'-ls-'+str(args.loss)
    print(run_name)
    # Extracting arguments from command line
    numberOfLayer=args.num_layers
    numberOfNeuronPerLayer=args.hidden_size
    numberOfNeuronOutputLayer=10
    activationFunction=args.activation
    initializer_type=args.weight_init
    eta=args.learning_rate
    regularize_coeef=args.weight_decay
    batch_size=args.batch_size
    optimizer=args.optimizer
    epoch=args.epochs
    loss=args.loss
    dataset=args.dataset
    confusion_bit=args.confution_matrix
    # Initializing Weights & Biases run
    wandb.init(project =args.project_name,entity=args.wandb_entity,name=run_name)
    # Loading dataset
    if dataset =='fashion_mnist':
      print('fashion_mnist_data')
      (train_image, train_class),(test_image, test_class) = fashion_mnist.load_data()
      train_image1=train_image.reshape(train_image.shape[0],-1)
      train_image_val=train_image1[int(0.9*train_image1.shape[0]):]
      train_class_val=train_class[int(0.9*train_image1.shape[0]):]
      train_image=train_image1[:int(0.9*train_image1.shape[0])]
      train_class=train_class[:int(0.9*train_image1.shape[0])]
      train_image=train_image/256
      train_image_val=train_image_val/256
      test_image1=test_image.reshape(test_image.shape[0],-1)
      test_image1=test_image1/256
    else:
      print('mnist_data')
      (train_image, train_class),(test_image, test_class) = mnist.load_data()
      train_image1=train_image.reshape(train_image.shape[0],-1)
      train_image_val=train_image1[int(0.9*train_image1.shape[0]):]
      train_class_val=train_class[int(0.9*train_image1.shape[0]):]
      train_image=train_image1[:int(0.9*train_image1.shape[0])]
      train_class=train_class[:int(0.9*train_image1.shape[0])]
      train_image=train_image/256
      train_image_val=train_image_val/256
      test_image1=test_image.reshape(test_image.shape[0],-1)
      test_image1=test_image1/256
      
    numberOfNeuronPrevLayer=train_image.shape[1]
    layer_objects=[]
    layer_objects_grad=[]
    # Creating layer objects and store in the array
    for i in range(numberOfLayer):
      if i ==numberOfLayer-1 :
        layer_object=layer(numberOfNeuronOutputLayer,numberOfNeuronPrevLayer,initializer_type,activationFunction)
        layer_objects.append(layer_object)
        layer_objects_grad.append(copy.deepcopy(layer_object))
      else:
        layer_object=layer(numberOfNeuronPerLayer,numberOfNeuronPrevLayer,initializer_type,activationFunction)
        layer_objects.append(layer_object)
        layer_objects_grad.append(copy.deepcopy(layer_object))
        numberOfNeuronPrevLayer=numberOfNeuronPerLayer
    # Initializing Neural Network object
    trainer=NeuralNetwork(train_image_val,train_class_val,loss)
    # Training using different optimizers depending on input
    if optimizer=='stochastic':
      layer_objects=trainer.schotastic_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='momentum':
      layer_objects=trainer.momentum_Gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='nesterov_accelerated':
      layer_objects=trainer.Nestrov_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='RmsProp':
      layer_objects=trainer.RmsProp(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='adam':
      layer_objects=trainer.Adam_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='nadam':
      layer_objects=trainer.nadam_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    test=TestingModel()
    output=test.CalculateTest(layer_objects,test_image1)  ## testing models with test data calculating loss and accuracy
    accuracy=test.zeroOneModel(output,test_class)
    test_loss=test.crossEntropyLoss(layer_objects,test_image1,test_class)
    print("Test Accuracy is :",accuracy)
    print("Test Loss is " ,test_loss)
    if confusion_bit ==1 :
      confusion_matrix_create(output,list(test_class))  #create confusion matrix on wandb
    wandb.finish()


if __name__ == "__main__":
  # Initialize the argument parser
  parser = argparse.ArgumentParser()
  # add arguments to the parsers
  parser.add_argument('-wp','--project_name',type=str,default='CS6910-Assignment',help='Project name used to track experiments in Weights & Biases dashboard')
  parser.add_argument('-we','--wandb_entity',type=str,default='amar_cs23m011',help='Project name used to track experiments in Weights & Biases dashboard')
  parser.add_argument('-d','--dataset',type=str,default='fashion_mnist',choices=['mnist', 'fashion_mnist'],help='choose any of the Datasets to use')
  parser.add_argument('-e','--epochs',type=int,default=15,help='Number of epochs to train neural network')
  parser.add_argument('-b','--batch_size',type=int,default=16,help='Batch size used to train neural network')
  parser.add_argument('-l','--loss',type=str,default='cross_entropy',choices=['cross_entropy', 'mean_squared_error'],help='loss function based on which we evaluate the model')
  parser.add_argument('-o','--optimizer',type=str,default='adam',choices=['stochastic', 'momentum','nesterov_accelerated','RmsProp','adam','nadam'],help='optimzer algorithm to evaluate the model')
  parser.add_argument('-a','--activation',type=str,default='relu',choices=['relu','sigmoid','tanh'],help='activation function used in the model')
  #parser.add_argument('-a','--activation',type=str,default='relu',choices=['relu','sigmoid','tanh'],help='activation function used in the model')
  parser.add_argument('-w_d','--weight_decay',type=float,default=0,help='Weight decay used by optimizers.')
  parser.add_argument('-w_i','--weight_init',type=str,default='xavier',choices=['random', 'xavier'],help='Weight initializer used by models.')
  parser.add_argument('-nhl','--num_layers',type=int,default=3,help='Number of hidden layers used in feedforward neural network')
  parser.add_argument('-sz','--hidden_size',type=int,default=128,help='Number of hidden layers used in feedforward neural network')
  parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help='Learning rate used to optimize model parameters')
  parser.add_argument('-cm','--confution_matrix',type=int,default=0,choices=[0,1],help='create confution matrix')
  # Parse the command-line arguments
  args = parser.parse_args()
  main(args)
  
