# RegNo: 2312089

import numpy as np

class Neuron:
    def __init__(self, weights, lambda_, delta_weights):
        self.lambda_ = lambda_
        self.value = 0
        self.weights = weights
        self.old_delta_weights = delta_weights

    # Returns the value of the activation function given a value
    def activateFunct(self, val):
        return 1.0 / (1.0 + np.exp(self.lambda_ * val * -1))
    
    # Sets the Neurons value
    def updateValue(self, val):
        self.value = self.activateFunct(val)

    # Sets the neurons weight
    def setWeights(self, weights):
        self.weights = weights

    # Updates the weights of the neuron and stores its old delta weights
    def updateWeights(self, delta_weights):
        for i in range(len(delta_weights)):
            self.weights[i] += delta_weights[i]
        self.old_delta_weights = delta_weights

    # Returns the value times weight for a specific connection between neurons
    def weightMult(self, index):
        return self.value * self.weights[index]