import numpy as np
import matplotlib.pyplot as plt
import copy

class ActivationFunctions:
    def sigmoid(self, u):
        x = []
        for i in u:
            if i > 200:
                x.append(1)
            elif i < -200:
                x.append(0)
            else:
                x.append(1 / (1 + np.exp(-i)))
        return np.array(x)

    def sigmoidGrad(self, u):
        f = self.sigmoid(u)
        return f * (1 - f)

    def tanh(self, u):
        return np.tanh(u)

    def tanhGrad(self, u):
        return 1 - np.square(np.tanh(u))

    def relu(self, u):
        return np.maximum(0, u)

    def reluGrad(self, u):
        return 1 * (u > 0)

    def softmax(self, u):
        x = np.exp(u - np.max(u))
        return x / np.sum(x)
