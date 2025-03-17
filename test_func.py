# Import necessary modules
from activation_func import ActivationFunctions
from layers import layer          # Importing layer class module
import numpy as np
import matplotlib.pyplot as plt
import copy
# Define a class for testing the model
class TestingModel:
  # Method for forward propagation through the layers
  def forwardPropagation1(self,layer_objects,input1):
    x=ActivationFunctions()
    # Iterate through each layer in the network
    for i in range(len(layer_objects)-1):
      if i==0:
        # Compute the activation and apply the activation function
        layer_objects[i].a=layer_objects[i].b+np.dot(layer_objects[i].w,input1)
        if layer_objects[i].activationFunction=='sigmoid':
          layer_objects[i].h=x.sigmoid(layer_objects[i].a)
        if layer_objects[i].activationFunction=='tanh':
          layer_objects[i].h=x.tanh(layer_objects[i].a)
        if layer_objects[i].activationFunction=='relu':
          layer_objects[i].h=x.relu(layer_objects[i].a)
      else:
        # For subsequent layers
        layer_objects[i].a=layer_objects[i].b+np.matmul(layer_objects[i].w,layer_objects[i-1].h)
        if layer_objects[i].activationFunction=='sigmoid':
          layer_objects[i].h=x.sigmoid(layer_objects[i].a)
        if layer_objects[i].activationFunction=='tanh':
          layer_objects[i].h=x.tanh(layer_objects[i].a)
        if layer_objects[i].activationFunction=='relu':
          layer_objects[i].h=x.relu(layer_objects[i].a)
    # Compute activation for the last layer using softmax
    layer_objects[len(layer_objects)-1].a=layer_objects[len(layer_objects)-1].b+np.dot(layer_objects[len(layer_objects)-1].w,layer_objects[len(layer_objects)-2].h)
    layer_objects[len(layer_objects)-1].h=x.softmax(layer_objects[len(layer_objects)-1].a)
    return layer_objects

  def zeroOneModel(self,Y_pred,Y_test):    # Method to calculate accuracy of the model
    return np.mean(Y_pred==Y_test)

  def CalculateTest(self,layer_object,test_image):  # Method to calculate predictions for test data
    output=[]
    for tt in test_image:
      ls=self.forwardPropagation1(layer_object,tt)
      output.append(np.argmax(ls[len(ls)-1].h))
    return output
  def crossEntropyLoss(self,layer_objects,test_image,test_class):   # Method to calculate cross-entropy loss
    ans=0
    for x,y in zip(test_image,test_class):
      ls=self.forwardPropagation1(layer_objects,x)
      #print(np.log(ls[len(ls)-1].h))
      ans=ans+(-1)*np.log(ls[len(ls)-1].h[y]+1e-8)
    return ans/test_class.shape[0]
  def squareError(self,layer_objects,test_image,test_class):     # Method to calculate mean square error
    ans=0
    for x,y in zip(test_image,test_class):
      ls=self.forwardPropagation1(layer_objects,x)
      #print(np.log(ls[len(ls)-1].h))
      one_hot=np.zeros(layer_objects[len(layer_objects)-1].numberOfNeuronPerLayer)
      one_hot[y]=1
      ans=ans+np.sum(np.square(ls[len(ls)-1].h-one_hot))
    return ans/test_class.shape[0]
  
