#import necessery modules
import numpy as np
import matplotlib.pyplot as plt
import copy
class layer:  #Initialize a neural network layer.
  def __init__(self,numberOfNeuronPerLayer,numberOfNeuronPrevLayer,initializer_type,activationFunction='None'):
    #initialize each variable in the class
    self.numberOfNeuronPerLayer=numberOfNeuronPerLayer
    self.activationFunction=activationFunction
    self.w=np.zeros(shape=(numberOfNeuronPerLayer,numberOfNeuronPrevLayer))
    self.gradw=np.zeros(shape=(numberOfNeuronPerLayer,numberOfNeuronPrevLayer))
    self.b=np.zeros(shape=(numberOfNeuronPerLayer))
    self.gradb=np.zeros(shape=(numberOfNeuronPerLayer))
    self.a=np.zeros(shape=(numberOfNeuronPerLayer))
    self.grad_a=np.zeros(shape=(numberOfNeuronPerLayer))
    self.h=np.zeros(shape=(numberOfNeuronPerLayer))
    self.grad_h=np.zeros(shape=(numberOfNeuronPerLayer))
    self.initialize_parameter(initializer_type)
  def initialize_gradient(self):       #Initialize gradients of weights, biases, activation, and hidden layers to zeros.
    self.gradw=np.zeros_like(self.gradw)
    self.gradb=np.zeros_like(self.gradb)
    self.grad_a=np.zeros_like(self.grad_a)
    self.grad_h=np.zeros_like(self.grad_h)

  def initialize_parameter(self,type='None'):
    #Initialize parameters (weights) of the layer based on the specified initializer type.
    #type (str): Type of weight initialization ('random', 'xavier', 'Xavier_Uniform').
    if type=='random':
      self.w=np.random.randn(self.w.shape[0],self.w.shape[1])
    elif type == 'xavier':
      self.w=np.random.normal(scale=np.sqrt(2/(self.w.shape[0]+self.w.shape[1])),size=(self.w.shape[0],self.w.shape[1]))
    elif type == 'Xavier_Uniform':
      self.w=np.random.uniform(low=-np.sqrt(6/(self.w.shape[0]+self.w.shape[1])), high=np.sqrt(6/(self.w.shape[0]+self.w.shape[1])),size=((self.w.shape[0],self.w.shape[1])))
