import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Generate Input Data (X) and Target Outputs (y)
X_train = np.linspace(0.1, 1, 20)  # 20 input training values between 0.1 and 1
X_test = np.linspace(0.1, 1, 50)  # 50 input testing values between 0.1 and 1

def true_function(X):
    return (1 + 0.6 * np.sin(2 * np.pi * X / 0.7) + 0.3 * np.sin(2 * np.pi * X)) / 2

y_train = true_function(X_train)
y_test = true_function(X_test)


# Hyperparameters
hidden_neurons = 7
learning_rate = 0.01
epochs = 10000

# Initializing Weights and Biases
np.random.seed(42)  # For reproducibility
input_to_hidden_weights = np.random.randn(1, hidden_neurons)
hidden_to_output_weights = np.random.randn(hidden_neurons, 1)
hidden_bias = np.random.randn(hidden_neurons)
output_bias = np.random.randn(1)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


training_errors = []

# Training loop
for epoch in range(1, epochs + 1):
    # Forward propagation
    hidden_input = np.dot(X_train.reshape(-1, 1), input_to_hidden_weights) + hidden_bias
    hidden_output = tanh(hidden_input)
    final_input = np.dot(hidden_output, hidden_to_output_weights) + output_bias
    final_output = final_input

    # Calculate error
    error = y_train - final_output.flatten()
    mse = np.mean(np.square(error))
    training_errors.append(mse)

    # Backpropagation
    d_output = -2 * error.reshape(-1, 1)  # Derivative of MSE
    d_hidden = d_output.dot(hidden_to_output_weights.T) * tanh_derivative(hidden_input)

    # Weight updates
    hidden_to_output_weights -= learning_rate * hidden_output.T.dot(d_output)
    output_bias -= learning_rate * d_output.sum(axis=0)
    input_to_hidden_weights -= learning_rate * X_train.reshape(-1, 1).T.dot(d_hidden)
    hidden_bias -= learning_rate * d_hidden.sum(axis=0)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Training Error: {mse:.6f}")

print("Training complete.")

# Generate predictions for training data
hidden_input_train = np.dot(X_train.reshape(-1, 1), input_to_hidden_weights) + hidden_bias
hidden_output_train = tanh(hidden_input_train)
final_output_train = (np.dot(hidden_output_train, hidden_to_output_weights) + output_bias).flatten()

# Generate predictions for testing data
hidden_input_test = np.dot(X_test.reshape(-1, 1), input_to_hidden_weights) + hidden_bias
hidden_output_test = tanh(hidden_input_test)
final_output_test = (np.dot(hidden_output_test, hidden_to_output_weights) + output_bias).flatten()

# Calculate test error
test_error = np.mean(np.square(y_test - final_output_test))
print(f"Test Error: {test_error:.6f}")

# Plotting
plt.figure(figsize=(15, 5))

# Training Error
plt.subplot(1, 2, 1)
plt.plot(range(0, epochs, epochs // 1000), training_errors[::epochs // 1000], 'b-', label='Training Error')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training Error over Epochs')
plt.legend()
plt.grid(True)

# Predictions vs Targets
plt.subplot(1, 2, 2)
# Plot training data and predictions
plt.plot(X_train, y_train, 'bo', alpha=0.5, label='Training Targets', markersize=4)
plt.plot(X_train, final_output_train, 'b-', label='Training Predictions', linewidth=2)

# Plot testing data and predictions
plt.plot(X_test, y_test, 'ro', alpha=0.5, label='Testing Targets', markersize=4)
plt.plot(X_test, final_output_test, 'r--', label='Testing Predictions', linewidth=2)

plt.xlabel('Input (X)')
plt.ylabel('Output (y)')
plt.title('Model Predictions vs Targets')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
