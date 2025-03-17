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
#760091de6b192857b226ee4bdecf4e7f93175087 
def confusion_matrix_create(pred,true_p):
  label_class=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
  wandb.log({ "Confusion Matrix" : wandb.sklearn.plot_confusion_matrix(true_p,pred,label_class)})
def main():
  if 2==2:
    numberOfLayer=3
    #with wandb.init(config=config, ):
    #config=wandb.config
    #wandb.run.name = 'bs-'+str(config.batch_size)+'-lr-'+ str(config.learning_rate)+'-ep-'+str(config.number_of_epochs)+ '-op-'+str(config.optimizer)+ '-nhl-'+str(config.number_of_hidden_layers)+'-shl-'+str(config.size_of_every_hidden_layer)+ '-act-'+str(config.activation_functions)+'-wd-'+str(config.weight_decay)+'-wi-'+str(config.weight_initialisation)
    numberOfNeuronPerLayer=128
    numberOfNeuronOutputLayer=10
    activationFunction='tanh'
    initializer_type='xavier'
    eta=0.01
    regularize_coeef=0
    batch_size=64
    optimizer='adam'
    epoch=15
    loss="cross_entropy"
    dataset='fashion_mnist'
    if dataset =='fashion_mnist':
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
      
    #test_image1=test_image.reshape(test_image.shape[0],-1)
    #test_image1=test_image1/256
    #train_image2=train_image1[0:1]
    #train_class2=train_class[0:1]
    numberOfNeuronPrevLayer=train_image.shape[1]
    layer_objects=[]
    layer_objects_grad=[]
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
    trainer=NeuralNetwork(train_image_val,train_class_val,loss)
    wandb.init(project ='CS6910-Assignment', magic=False,reinit = True)
    layer_objects=trainer.Adam_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    #print(layer_objects[len(layer_objects)-1].h)
    test=TestingModel()
    output=test.CalculateTest(layer_objects,test_image1)
    accuracy=test.zeroOneModel(output,test_class)
    test_loss=test.crossEntropyLoss(layer_objects,test_image1,test_class)
    print("Test Accuracy is :",accuracy)
    print("Test Loss is " ,test_loss)
    #wandb.init(project ='CS6910-Assignment', magic=True,reinit = True)
    confusion_matrix_create(output,list(test_class))
	


if __name__ == "__main__":
  main()
