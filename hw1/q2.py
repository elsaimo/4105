import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to compute predictions for multivariate regression
def predict_multivariate(X, theta):
    return np.dot(X, theta)

# Function to compute the cost for multivariate regression
def compute_cost_multivariate(X, y, theta):
    m = len(y)
    predictions = predict_multivariate(X, theta)
    return (1/(2*m)) * np.sum((predictions - y)**2)

# Function to perform gradient descent for multivariate regression
def gradient_descent_multivariate(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    X = np.insert(X, 0, 1, axis=1)  # Add a column of ones to X for the intercept term

    for i in range(iterations):
        predictions = predict_multivariate(X, theta)
        errors = predictions - y
        for j in range(len(theta)):
            theta[j] = theta[j] - (learning_rate/m) * np.sum(errors * X[:, j])
        cost_history[i] = compute_cost_multivariate(X, y, theta)

    return theta, cost_history

# Load the dataset
file_path = '/mnt/data/D3.csv'
data = pd.read_csv(file_path)

# Prepare data
X = data[['X1', 'X2', 'X3']].values
y = data['Y'].values
theta_initial = np.zeros(X.shape[1] + 1)  # +1 for the intercept term

# Train the model with multiple explanatory variables
learning_rates = [0.1, 0.01]
results_multivariate = {}

for lr in learning_rates:
    theta, cost_history = gradient_descent_multivariate(X, y, theta_initial.copy(), lr, 1000)
    results_multivariate[lr] = {'theta': theta, 'cost_history': cost_history}

    # Plot the loss over iterations
    plt.plot(range(1000), cost_history, label=f'LR={lr}')

plt.title('Loss over Iterations for Multiple Explanatory Variables')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Select the best learning rate based on the final loss
best_lr = min(results_multivariate, key=lambda lr: results_multivariate[lr]['cost_history'][-1])
best_theta = results_multivariate[best_lr]['theta']
print(f"Best learning rate: {best_lr}, Best theta: {best_theta}")

# Predict function for new values
def predict_values(theta, X_new):
    X_new = np.insert(X_new, 0, 1, axis=1)  # Add intercept term
    return predict_multivariate(X_new, theta)

# New sets of (X1, X2, X3) values
X_new_values = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])

# Predictions
predictions = predict_values(best_theta, X_new_values)
print(f"Predictions for new values: {predictions}")
