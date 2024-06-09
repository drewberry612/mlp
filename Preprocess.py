# RegNo: 2312089

import pandas as pd
import numpy as np

if __name__ == "__main__":
    # If the csv's aren't being read properly, change the file path
    df = pd.read_csv('../Assignment Code/ce889_dataCollection.csv')

    # Finds the min and max of x, y, x_vel, y_vel for normalisation in the NeuralNetHolder class
    x_min = min(df.iloc[0])
    x_max = max(df.iloc[0])
    y_min = min(df.iloc[1])
    y_max = max(df.iloc[1])
    x_vel_min = min(df.iloc[4])
    x_vel_max = max(df.iloc[4])
    y_vel_min = min(df.iloc[3])
    y_vel_max = max(df.iloc[3])
    array = [x_min, x_max, y_min, y_max, x_vel_min, x_vel_max, y_vel_min, y_vel_max]
    print(array)

    # Saves these values to a numpy file
    np.save(f'minmax.npy', array)

    # Normalise the data and output to a training csv
    normalized_df=(df-df.min())/(df.max()-df.min())
    normalized_df.to_csv("../Assignment Code/training_data.csv", header=None, index=None)