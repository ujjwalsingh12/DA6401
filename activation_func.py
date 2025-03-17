# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import copy
# Define a class for activation functions
class ActivationFunctions:
  def sigmoid(self,u):         # Sigmoid activation function
    x=[]
    for i in u:
      if i>200:
        x.append(1)
      elif i< -200:
        x.append(0)
      else :
        x.append(1/(1+np.exp(-i)))
    return np.array(x)


  def sigmoidGrad(self,u):           # Gradient of sigmoid activation function
    f=self.sigmoid(u)
    return f*(1-f)

  def tanh(self,u):                  # Hyperbolic tangent (tanh) activation function
    return np.tanh(u)

  def tanhGrad(self,u):              # Gradient of tanh activation function
    return (1-np.square(np.tanh(u)))

  def relu(self,u):                  # Rectified Linear Unit (ReLU) activation function
    return np.maximum(0,u)

  def reluGrad(self,u):              # Gradient of ReLU activation function
    return 1*(u>0)

  def softmax(self,u):               # Softmax activation function
    x=np.exp(u-np.max(u))
    return x/np.sum(x)
