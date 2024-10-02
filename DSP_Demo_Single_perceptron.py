'''
DSP DEEP LEARNING 1: DEMO 1
NEURAL NET FROM SCRATCH 
OPTIMIZER: SGD WITH MOMENTUM
AUTHOR: Christian Dave T. Navesis
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.v_w = 0  # Momentum for weights
        self.v_b = 0  # Momentum for bias
    
    def forward(self, X):
        # Forward propagation
        Z = np.dot(X, self.weights) + self.bias
        return Z

    def mean_squared_error_loss(self, y_true, y_pred):
        # Mean squared error loss
        return np.mean((y_true - y_pred) ** 2)

    def train(self, X, y_true, learning_rate=0.01, momentum=0.9, epochs=4000):
        # Training the network using stochastic gradient descent with momentum
        for epoch in range(epochs):
            # Forward propagation
            predictions = self.forward(X)
            
            # Compute loss
            loss = self.mean_squared_error_loss(y_true, predictions)
            
            # Backpropagation
            dZ = (predictions - y_true) / len(X)
            dW = np.dot(X.T, dZ)
            db = np.sum(dZ)
            
            # Update momentum
            self.v_w = momentum * self.v_w - learning_rate * dW
            self.v_b = momentum * self.v_b - learning_rate * db
            
            # Update weights and bias
            self.weights += self.v_w
            self.bias += self.v_b
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def print_weights(self):
        print(f"Final weights:{self.weights}")

    def accuracy(self, X, y_true):
        predictions = self.forward(X)
        accuracy = np.mean(np.isclose(predictions, y_true, atol=1e-2))
        return accuracy

# Load data from Excel file
data = pd.read_excel("db.xlsx", usecols="B:G")
X = data.iloc[:, :-1].values / 100  # Normalize input data by dividing by 100
y = data.iloc[:, -1].values / 100   # Normalize output data by dividing by 100

# Initialize and train the neural network
nn = NeuralNetwork(input_size=X.shape[1])
nn.train(X, y)

# Print final weights
true_weights = np.array([0.4, 0.15, 0.15, 0.25, 0.05])

# Generate test data
X_test = np.random.rand(100, 5)
y_test = np.dot(X_test, true_weights)

# Calculate accuracy on test data
test_accuracy = nn.accuracy(X_test, y_test)
print("========================================")
nn.print_weights()
print(f"True weights:{true_weights}")
print(f"Test accuracy: {test_accuracy}")
