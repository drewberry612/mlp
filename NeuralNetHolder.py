# RegNo: 2312089

import Network
import numpy as np

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

        # Load all best weights from numpy file
        array = np.load("weights.npz")
        inp = array["input"]
        hidden = array["hidden"]

        # Load the minimum and maximum values of x and y in the csv file
        array = np.load("minmax.npy")
        self.x_min = array[0]
        self.x_max = array[1]
        self.y_min = array[2]
        self.y_max = array[3]
        self.x_vel_min = array[4]
        self.x_vel_max = array[5]
        self.y_vel_min = array[6]
        self.y_vel_max = array[7]

        # Initialise the network and set the best weights
        self.network = Network.Network(0.095, 8, 0.0375) # These values are the optimum parameters
        self.network.initialiseNetwork()
        self.network.setWeights(inp, hidden)
    
    def predict(self, input_row):
        input_row = input_row.split(",")
        inputs = []
        
        # Normalise the input row
        inputs.append((float(input_row[0]) - self.x_min) / (self.x_max - self.x_min))
        inputs.append((float(input_row[1]) - self.y_min) / (self.y_max - self.y_min))

        # Perform one pass of the network
        self.network.feedForward(inputs)

        # Get output values
        x = self.network.layers[2][0].value
        y = self.network.layers[2][1].value

        # Denormalisation
        x_velocity = x * (self.x_vel_max - self.x_vel_min) + self.x_vel_min
        y_velocity = y * (self.y_vel_max - self.y_vel_min) + self.y_vel_min
        return y_velocity, x_velocity
