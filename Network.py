# RegNo: 2312089

import Neuron
import random
import numpy as np

class Network:
    def __init__(self, eta, hidden_neurons, alpha):
        self.layers = []

        # Hyper parameters
        self.lambda_ = 0.8 # fixed
        self.eta = eta
        self.hidden_neurons = hidden_neurons
        self.alpha = alpha


    # For use in NeuralNetHolder
    # Sets the input and hidden neuron weights to the given parameters
    def setWeights(self, inp_weights, hidden_weights):
        for i in range(len(self.layers[0])):
            self.layers[0][i].setWeights(inp_weights[i])

        for i in range(self.hidden_neurons):
            self.layers[1][i].setWeights(hidden_weights[i])


    # Creates the network
    def initialiseNetwork(self):
        # Input Layer
        newLayer = []
        for i in range(2): # For every input, create a neuron
            weights = []
            for j in range(self.hidden_neurons): # Get random weights for each neuron in the next layer
                weights.append(random.uniform(-1,1))
            newLayer.append(Neuron.Neuron(weights, self.lambda_, [0] * 2)) # Creates an array of 0's for old_delta_weights
        self.layers.append(newLayer)


        # Hidden Layer
        newLayer = []
        for i in range(self.hidden_neurons): # For every hidden_neuron hyper param, create a neuron
            weights = []
            for j in range(2): # Get random weights for each neuron in the next layer
                weights.append(random.uniform(-1,1))
            newLayer.append(Neuron.Neuron(weights, self.lambda_, [0] * self.hidden_neurons)) # Creates an array of 0's for old_delta_weights
        self.layers.append(newLayer)


        # Output Layer
        newLayer = []
        for j in range(2): # No need to initialise weights
            newLayer.append(Neuron.Neuron([], self.lambda_, [0] * 2))
        self.layers.append(newLayer)


    # Performs one feed forward pass
    def feedForward(self, inputs):
        # Input layer
        for i in range(2): # Sets the input neuron value to the input
            self.layers[0][i].value = inputs[i]

        # Hidden Layer
        for h in range(self.hidden_neurons):
            weightedSum = 0
            for inp in self.layers[0]: # Sums the weights times values of each neuron connected to the hidden neuron
                weightedSum += inp.weightMult(h)
            self.layers[1][h].updateValue(weightedSum)

        # Output Layer
        for o in range(2):
            weightedSum = 0
            for hidden in self.layers[1]: # Sums the weights times values of each neuron connected to the output neuron
                weightedSum += hidden.weightMult(o)
            self.layers[2][o].updateValue(weightedSum)
    

    # Performs one pass of backpropogation
    def backpropogation(self, actual_outputs):
        # Error calculations
        observed_outputs = [self.layers[2][0].value, self.layers[2][1].value]
        errors = []
        for i in range(len(actual_outputs)):
            errors.append(observed_outputs[i] - actual_outputs[i])

        # Finding local gradients for output layer
        local_gradients = []
        for i in range(len(actual_outputs)):
            local_gradients.append(self.lambda_ * observed_outputs[i] * (1 - observed_outputs[i]) * errors[i])

        # Updating of weights in output layer
        for hidden in self.layers[1]:
            delta_weights = []
            weight_gradient_sum = 0 # Used in the calculation of gradient in the hidden layer
            for i in range(len(local_gradients)):
                delta_weights.append((self.eta * local_gradients[i] * hidden.value) + (self.alpha * hidden.old_delta_weights[i]))
                weight_gradient_sum += (local_gradients[i] * hidden.weights[i])
            hidden.updateWeights(delta_weights)

            # Finding local gradients for hidden layer
            local_gradients = []
            for i in range(len(actual_outputs)):
                local_gradients.append(self.lambda_ * hidden.value * (1 - hidden.value) * weight_gradient_sum)
        
        # Updating of weights in hidden layer
        for inp in self.layers[0]:
            delta_weights = []
            for i in range(len(local_gradients)):
                delta_weights.append((self.eta * local_gradients[i] * inp.value) + (self.alpha * inp.old_delta_weights[i]))
            inp.updateWeights(delta_weights)
        
        return errors
    

    # Performs one pass of validation
    def validation(self, inputs, predicted_outputs):
        self.feedForward(inputs)
        observed_outputs = [self.layers[2][0].value, self.layers[2][1].value]
        errors = []
        for i in range(len(predicted_outputs)):
            errors.append(observed_outputs[i] - predicted_outputs[i])

        return errors


    # Outputs the best weights to a numpy file for the NeuralNetHolder to access later
    def outputWeights(self):
        input_weights = []
        for i in self.layers[0]:
            input_weights.append(i.weights)
        
        hidden_weights = []
        for h in self.layers[1]:
            hidden_weights.append(h.weights)

        np.savez(f'weights.npz', input=input_weights, hidden=hidden_weights)