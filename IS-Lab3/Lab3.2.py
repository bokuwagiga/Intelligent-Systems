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
learning_rate_weights = 0.01
learning_rate_centers = 0.001
learning_rate_radius = 0.001
epochs = 10000

# Initialize weights randomly
np.random.seed(42)
b = np.random.random()
w1 = np.random.random()
w2 = np.random.random()

errors = []
c1_history = []
c2_history = []
r1_history = []
r2_history = []

for epoch in range(epochs):
    abs_error = 0
    # Accumulate gradients
    dc1, dc2 = 0, 0
    dr1, dr2 = 0, 0
    db, dw1, dw2 = 0, 0, 0

    for i in range(len(X_train)):
        # Forward pass
        rbf1 = gaussian_rbf(X_train[i], c1, r1)
        rbf2 = gaussian_rbf(X_train[i], c2, r2)
        y_pred = b + w1 * rbf1 + w2 * rbf2

        # Calculate error
        error = y_train[i] - y_pred
        abs_error += error ** 2

        # Gradient calculations
        # For centers
        dc1 += error * w1 * rbf1 * (X_train[i] - c1) / (r1 ** 2)
        dc2 += error * w2 * rbf2 * (X_train[i] - c2) / (r2 ** 2)

        # For radius
        dr1 += error * w1 * rbf1 * ((X_train[i] - c1) ** 2) / (r1 ** 3)
        dr2 += error * w2 * rbf2 * ((X_train[i] - c2) ** 2) / (r2 ** 3)

        # For weights and bias
        db += error
        dw1 += error * rbf1
        dw2 += error * rbf2

    # Update parameters
    mse = abs_error / len(X_train)

    # Update centers
    c1 += learning_rate_centers * dc1
    c2 += learning_rate_centers * dc2

    # Update radius (ensure they stay positive)
    r1 = max(0.01, r1 + learning_rate_radius * dr1)
    r2 = max(0.01, r2 + learning_rate_radius * dr2)

    # Update weights and bias
    b += learning_rate_weights * db
    w1 += learning_rate_weights * dw1
    w2 += learning_rate_weights * dw2

    # Store history
    errors.append(mse)
    c1_history.append(c1)
    c2_history.append(c2)
    r1_history.append(r1)
    r2_history.append(r2)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.6f}")
        print(f"Centers: c1={c1:.3f}, c2={c2:.3f}")
        print(f"Radius: r1={r1:.3f}, r2={r2:.3f}")
        print(f"Weights: b={b:.3f}, w1={w1:.3f}, w2={w2:.3f}\n")

# Generate predictions for test data
rbf1_test = gaussian_rbf(X_test, c1, r1)
rbf2_test = gaussian_rbf(X_test, c2, r2)
y_pred_test = b + w1 * rbf1_test + w2 * rbf2_test

# Create subplots
plt.figure(figsize=(15, 5))

# Centers evolution plot
plt.subplot(1, 2, 1)
plt.plot(c1_history, label='Center 1')
plt.plot(c2_history, label='Center 2')
plt.xlabel('Epoch')
plt.ylabel('Center Position')
plt.title('Evolution of Centers')
plt.legend()
plt.grid(True)
# radius evolution plot
plt.subplot(1, 2, 2)
plt.plot(r1_history, label='Radius 1')
plt.plot(r2_history, label='Radius 2')
plt.xlabel('Epoch')
plt.ylabel('Radius Value')
plt.title('Evolution of Radius')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Training error plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Error over Epochs')
plt.grid(True)
# Final predictions plot
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
print(f"Centers: c1={c1:.4f}, c2={c2:.4f}")
print(f"Radius: r1={r1:.4f}, r2={r2:.4f}")
print(f"Weights: b={b:.4f}, w1={w1:.4f}, w2={w2:.4f}")