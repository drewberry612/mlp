# RegNo: 2312089

import Network
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

class Training:
    def __init__(self):
        self.network = None
        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []

        # Loads the data from the csv file
        self.df = pd.read_csv("../Assignment Code/training_data.csv")
        self.shuffleData()


    # Shuffles the data frame to select a new permutation every epoch
    # The commented lines in this function are used for grid search
    # This is because I used half the data to perform the grid search to save time
    def shuffleData(self):
        # Randomly shuffles the data frame
        self.df = self.df.sample(frac=1, random_state=1)
        data = self.df.to_numpy()

        # These are the index of the end of the split
        split = math.ceil(len(data) * 0.75)
        #split1 = math.ceil(len(data) * 0.4)
        #split2 = math.ceil(len(data) * 0.5)

        # Reset arrays
        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []
        for i in range(len(data)):
            if i < split:
            #if i < split1:
                self.x_train.append([float(data[i][0]), float(data[i][1])])
                self.y_train.append([float(data[i][2]), float(data[i][3])])
            #elif i < split2:
            else:
                self.x_valid.append([float(data[i][0]), float(data[i][1])])
                self.y_valid.append([float(data[i][2]), float(data[i][3])])


    # Uses the same validation for every epoch but reorder the training data
    def shuffleTraining(self):
        # Creates a random list of indexes
        indexes = np.arange(0, len(self.x_train))
        np.random.shuffle(indexes)

        new_x = []
        new_y = []
        for i in indexes:
            new_x.append(self.x_train[i])
            new_y.append(self.y_train[i])

        self.x_train = new_x
        self.y_train = new_y


    # Runs one epoch
    def epoch(self):
        self.shuffleData()
        #self.shuffleTraining()

        train_errors = []
        valid_errors = []
        # Completes the training and collect the errors
        for i in range(len(self.x_train)):
            self.network.feedForward(self.x_train[i])
            train_errors.append(self.network.backpropogation(self.y_train[i]))

        # Completes the validation and collect the errors
        for i in range(len(self.x_valid)):
            valid_errors.append(self.network.validation(self.x_valid[i], self.y_valid[i]))
        
        # Finds the mean of the outputs for use in rmse calculation
        mean_train = np.mean(self.y_train)
        mean_valid = np.mean(self.y_valid)

        # Calculate the 4 rmse's that were found in the epoch
        train_y1_rmse = self.rmse(train_errors, mean_train, 0)
        train_y2_rmse = self.rmse(train_errors, mean_train, 1)
        valid_y1_rmse = self.rmse(valid_errors, mean_valid, 0)
        valid_y2_rmse = self.rmse(train_errors, mean_valid, 1)

        # Returns the average of these errors
        return (train_y1_rmse + train_y2_rmse) / 2, (valid_y1_rmse + valid_y2_rmse) / 2
    

    # RMSE calculation
    def rmse(self, errors, mean, index):
        rmse = 0
        for i in errors:
            rmse += (i[index] - mean)** 2
        rmse /= len(errors)
        rmse = math.sqrt(rmse)
        return rmse
    

    # Displays the graph for the whole training process
    def graph(self, training_errors, validation_errors):
        train = []
        valid = []
        for i in range(len(training_errors)):
            train.append([i+1, training_errors[i]])
            valid.append([i+1, validation_errors[i]])

        # Plots the epochs along the x axis and the RMSE along the y axis
        t_df = pd.DataFrame(train, columns=["Epoch", "RMSE"])
        v_df = pd.DataFrame(valid, columns=["Epoch", "RMSE"])
        sns.lineplot(x="Epoch", y="RMSE", data=t_df)
        sns.lineplot(x="Epoch", y="RMSE", data=v_df)
        plt.show()
        

    # Performs the whole grid search
    # Replaces the main() function when wanting to grid search
    def gridSearch(self):
        # first grid search
        # had no best params at the time
        #eta = {0.005, 0.01, 0.05, 0.1, 0.2}
        #hidden_neurons = {4, 8, 10, 12}
        #alpha = {0.005, 0.01, 0.05, 0.1, 0.2}

        # second grid search
        # at the time best params were 0.1, 8, 0.05
        #eta = {0.075, 0.1, 0.125}
        #hidden_neurons = {7, 8, 9}
        #alpha = {0.04, 0.05, 0.06}

        # third grid search
        # at the time best params were 0.1, 8, 0.05
        eta = {0.09, 0.095, 0.1, 0.105, 0.11}
        hidden_neurons = {8}
        alpha = {0.035, 0.0375, 0.04, 0.0425, 0.045}


        best_overall_rmse = 1
        best_params = {}
        iteration_count = 0

        # For every permutation, test the performance
        for e in eta:
            for h in hidden_neurons:
                for a in alpha:
                    iteration_count += 1 # Used for progress tracking
                    print(iteration_count)

                    # Initialise the network with the hyper params
                    self.network = Network.Network(e, h, a)
                    self.network.initialiseNetwork()

                    best_rmse = 1
                    count = 0
                    # Perform a max of 50 epochs with an early stopping tolerance of 5 epochs
                    for i in range(50):
                        train_rmse, valid_rmse = self.epoch()
                        if valid_rmse < best_rmse:
                            best_rmse = valid_rmse
                            count = 0
                        else: # If the epoch hasnt found a new minimum validation rmse, then add to the early stop counter
                            count += 1
                            if count == 5:
                                print("Stopping early...")
                                break
                    
                    # When training is complete, if the best RMSE is better than any found so far, a new best set of parameters was found
                    if best_rmse < best_overall_rmse:
                        best_overall_rmse = best_rmse
                        best_params = {e, h, a}
                        print("new best params found")

        # Output the params so that they can be pasted in main()
        print()
        print(best_overall_rmse)
        print(best_params)


    # Main function of the program
    # Performs the training of the network
    def main(self):
        # Initialise the network with random weights and given hyper params
        self.network = Network.Network(0.095, 8, 0.0375)
        self.network.initialiseNetwork()

        training_errors = []
        validation_errors = []
        best_rmse = 1
        count = 0
        epochs = 100
        for i in range(epochs): # Run 100 epochs
            train_rmse, valid_rmse = self.epoch()
            if valid_rmse < best_rmse:
                best_rmse = valid_rmse
                count = 0
            else: # If the epoch hasnt found a new minimum validation rmse, then add to the early stop counter
                count += 1
                if count == 10:
                    print("Stopping early...")
                    break
            print("Epoch: " + str(i) + "    Train RMSE: " + str(round(train_rmse, 5)) + "     Valid RMSE: " + str(round(valid_rmse, 5)))
            print()
            training_errors.append(train_rmse)
            validation_errors.append(valid_rmse)

        # Display the graph and output the best weights
        self.graph(training_errors, validation_errors)
        self.network.outputWeights()


# Comment out main() and uncomment gridSearch() to perform grid search
# and vice versa to perform training
if __name__ == "__main__":
    t = Training()
    #t.gridSearch()
    t.main()