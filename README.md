# 🌑 Lunar Lander MLP

This project involved creating a **Multi-layer Perceptron (MLP)** from scratch in Python. The goal was to train the MLP to fly a lunar lander game using the x and y positions relative to the target as inputs, and the lander's thruster and turning as outputs.

## ⚙️ Key Features

- **Data Collection**: I collected all the training data myself, ensuring the dataset was wide and even to improve model generalisation.

- **Neural Network Architecture**: The model uses **feed-forward** and **backpropagation** with a **sigmoid activation function** and a learning rate (alpha) greater than zero.

- **Hyperparameter Tuning**: A **grid search** was performed on the hyperparameters to determine the best set for this specific data, ensuring optimal model performance.

- **Evaluation**: The model's performance was evaluated using **Root Mean Squared Error (RMSE)**, which provided insights into the overall accuracy of the model's predictions.

- **Early Stopping**: The model was trained for roughly **100 epochs**, with an **early stopping clause** to halt training if there was no improvement in validation RMSE for 5 consecutive epochs.

This repository includes a **presentation** detailing the implementation, research, and results of the project.

*Note: The lunar lander game itself is not included in this repository, but the MLP is fully trained and evaluated based on the dataset collected for the task.*
