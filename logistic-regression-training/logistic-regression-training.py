
import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=1000):

    # Step 1: Convert input into NumPy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Step 2: Get number of samples and features
    N, D = X.shape

    # Step 3: Initialize weights and bias
    w = np.zeros(D)
    b = 0.0

    # Step 4: Training loop
    for _ in range(steps):

        # Calculate predictions
        z = X @ w + b
        p = _sigmoid(z)

        # Calculate prediction error
        error = p - y

        # Calculate gradients
        dw = (X.T @ error) / N
        db = np.sum(error) / N

        # Update weights and bias
        w = w - lr * dw
        b = b - lr * db

    # Step 5: Return trained parameters
    return w, float(b)
