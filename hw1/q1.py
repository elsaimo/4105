import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to compute the hypothesis / predictions
def predict(X, theta):
    return theta[0] + theta[1] * X

# Function to compute the cost of a given prediction
def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    return (1/(2*m)) * np.sum((predictions - y)**2)

# Function to perform gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = predict(X, theta)
        theta[0] = theta[0] - (learning_rate/m) * np.sum(predictions - y)
        theta[1] = theta[1] - (learning_rate/m) * np.sum((predictions - y) * X)
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

# Load the dataset
file_path = '/mnt/data/D3.csv'
data = pd.read_csv(file_path)

# Initialize parameters
theta_initial = [0, 0]
iterations = 1000
learning_rates = [0.1, 0.01]

# Dictionary to store results for each variable
results = {}

# Train the model for each explanatory variable
for column in ['X1', 'X2', 'X3']:
    X = data[column]
    y = data['Y']
    results[column] = {}

    for lr in learning_rates:
        theta, cost_history = gradient_descent(X, y, theta_initial.copy(), lr, iterations)
        results[column][lr] = {'theta': theta, 'cost_history': cost_history}

        # Plotting the final regression model and loss over iteration for this variable and learning rate
        plt.figure(figsize=(14, 6))

        # Plot the regression line
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, color='blue', label='Data Points')
        plt.plot(X, predict(X, theta), color='red', label='Regression Line')
        plt.title(f'Regression Line for {column} with LR={lr}')
        plt.xlabel(column)
        plt.ylabel('Y')
        plt.legend()

        # Plot the loss over iterations
        plt.subplot(1, 2, 2)
        plt.plot(range(iterations), cost_history, color='green')
        plt.title(f'Loss over Iterations for {column} with LR={lr}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

# Find the variable with the lowest final loss
lowest_loss = float('inf')
best_variable = None
best_lr = None

for column in results:
    for lr, details in results[column].items():
        final_loss = details['cost_history'][-1]
        if final_loss < lowest_loss:
            lowest_loss = final_loss
            best_variable = column
            best_lr = lr

print(f"Best variable: {best_variable}, Learning Rate: {best_lr}, Lowest Loss: {lowest_loss}")
