import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('TkAgg')

# Generate 2D input data
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, y)

# Generate target function for example: z = sin(pi*x) * cos(pi*y)
Z = np.sin(np.pi * X) * np.cos(np.pi * Y)

# Prepare data for training
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

# Combine inputs
inputs = np.column_stack((X_flat, Y_flat))

# Normalize the target values to [-1, 1] range
Z_min, Z_max = Z_flat.min(), Z_flat.max()
Z_flat_normalized = 2 * (Z_flat - Z_min) / (Z_max - Z_min) - 1

# Split data (80% training, 20% testing)
indices = np.random.permutation(len(inputs))
split_idx = int(0.8 * len(inputs))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = inputs[train_indices]
y_train = Z_flat_normalized[train_indices]
X_test = inputs[test_indices]
y_test = Z_flat_normalized[test_indices]

# Hyperparameters
hidden_neurons = 8
learning_rate = 0.001
epochs = 10000
regularization = 0.0001  # L2 regularization parameter
clip_value = 1.0

# Initializing Weights and Biases
np.random.seed(42)
input_to_hidden_weights = np.random.randn(2, hidden_neurons) * 0.1
hidden_to_output_weights = np.random.randn(hidden_neurons, 1) * 0.1
hidden_bias = np.random.randn(hidden_neurons) * 0.1
output_bias = np.random.randn(1) * 0.1

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def clip_gradients(gradients, threshold):
    norm = np.linalg.norm(gradients)
    if norm > threshold:
        gradients = gradients * (threshold / norm)
    return gradients

training_errors = []

# Training loop
for epoch in range(1, epochs + 1):
    # Forward propagation
    hidden_input = np.dot(X_train, input_to_hidden_weights) + hidden_bias
    hidden_output = tanh(hidden_input)
    final_input = np.dot(hidden_output, hidden_to_output_weights) + output_bias
    final_output = final_input

    # Calculate error with L2 regularization
    error = y_train - final_output.flatten()
    l2_reg = (regularization/2) * (np.sum(np.square(input_to_hidden_weights)) +
                                  np.sum(np.square(hidden_to_output_weights)))
    mse = np.mean(np.square(error)) + l2_reg
    training_errors.append(mse)

    # Backpropagation
    d_output = -2 * error.reshape(-1, 1)  # Derivative of MSE
    d_hidden = d_output.dot(hidden_to_output_weights.T) * tanh_derivative(hidden_input)

    # Calculate gradients with regularization
    d_hidden_to_output = hidden_output.T.dot(d_output) + regularization * hidden_to_output_weights
    d_input_to_hidden = X_train.T.dot(d_hidden) + regularization * input_to_hidden_weights

    # Clip gradients
    d_hidden_to_output = clip_gradients(d_hidden_to_output, clip_value)
    d_input_to_hidden = clip_gradients(d_input_to_hidden, clip_value)

    # Weight updates
    hidden_to_output_weights -= learning_rate * d_hidden_to_output
    output_bias -= learning_rate * d_output.sum(axis=0)
    input_to_hidden_weights -= learning_rate * d_input_to_hidden
    hidden_bias -= learning_rate * d_hidden.sum(axis=0)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Training Error: {mse:.6f}")

# Generate predictions for the entire surface
def predict(X_input):
    hidden = tanh(np.dot(X_input, input_to_hidden_weights) + hidden_bias)
    return np.dot(hidden, hidden_to_output_weights) + output_bias

# Get predictions and denormalize
predictions = predict(inputs)
Z_pred = predictions.reshape(X.shape)
Z_pred = (Z_pred + 1) * (Z_max - Z_min) / 2 + Z_min

# Plotting
fig = plt.figure(figsize=(15, 5))

# Original Surface
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Original Surface')
fig.colorbar(surf1)

# Predicted Surface
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_pred, cmap='viridis')
ax2.set_title('Predicted Surface')
fig.colorbar(surf2)

plt.tight_layout()
plt.show()

# Calculate test error (using normalized values)
test_predictions = predict(X_test)
test_error = np.mean(np.square(y_test - test_predictions.flatten()))
print(f"Test Error: {test_error:.6f}")