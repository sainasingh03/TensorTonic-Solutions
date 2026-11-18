
import numpy as np

def rmsprop_step(w, g, s, lr=0.01, beta=0.9, eps=1e-8):

    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    # Step 1: Update running squared gradient
    new_s = beta * s + (1 - beta) * (g ** 2)

    # Step 2: Update parameters
    new_w = w - lr * g / np.sqrt(new_s + eps)

    # Round results to 6 decimal places
    new_w = np.round(new_w, 6)
    new_s = np.round(new_s, 6)

    return new_w, new_s
