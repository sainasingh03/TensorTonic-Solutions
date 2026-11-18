
import numpy as np

def dropout(x, p, rng=None):
    # Convert input into NumPy array
    x = np.asarray(x, dtype=float)

    # Generate random values
    if rng is not None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)

    # Keep elements with probability (1 - p)
    mask = (random_values >= p).astype(float)

    # Inverted dropout scaling
    dropout_pattern = mask / (1 - p)

    # Apply dropout
    output = x * dropout_pattern

    return output, dropout_pattern
