import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
# Generate x points
X_train = np.linspace(0.1, 1, 20)
X_test = np.linspace(0.1, 1, 50)


def true_function(x):
    return (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2


# Generate target values
y_train = true_function(X_train)
y_test = true_function(X_test)

# RBF parameters (manually selected)
c1, r1 = 0.3, 0.2
c2, r2 = 0.7, 0.2

def gaussian_rbf(x, c, r):
    return np.exp(-(x - c) ** 2 / (2 * r ** 2))


rbf1_train = gaussian_rbf(X_train, c1, r1)
rbf2_train = gaussian_rbf(X_train, c2, r2)

# Training parameters
learning_rate = 0.01
epochs = 10000

# Initialize weights randomly
np.random.seed(42)
b = np.random.random()
w1 = np.random.random()
w2 = np.random.random()

errors = []

for epoch in range(epochs):
    abs_error = 0

    for i in range(len(X_train)):
        y_pred = b + w1 * rbf1_train[i] + w2 * rbf2_train[i]
        error = y_train[i] - y_pred
        abs_error += error ** 2

        b += learning_rate * error
        w1 += learning_rate * error * rbf1_train[i]
        w2 += learning_rate * error * rbf2_train[i]

    mse = abs_error / len(X_train)
    errors.append(mse)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.6f}")
        print(f"Weights: b={b:.3f}, w1={w1:.3f}, w2={w2:.3f}\n")

# Generate predictions for test data
rbf1_test = gaussian_rbf(X_test, c1, r1)
rbf2_test = gaussian_rbf(X_test, c2, r2)
y_pred_test = b + w1 * rbf1_test + w2 * rbf2_test

# Plotting results
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Error over Epochs')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, c='blue', label='Training Data')
plt.plot(X_test, y_test, 'g-', label='True Function')
plt.plot(X_test, y_pred_test, 'r--', label='RBF Prediction')
c1_y, c2_y = true_function(c1), true_function(c2)
plt.scatter([c1, c2], [c1_y, c2_y], c='black', marker='x',
            s=100, label='RBF Centers')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('RBF Network Approximation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nFinal parameters:")
print(f"Weights: b={b:.4f}, w1={w1:.4f}, w2={w2:.4f}")