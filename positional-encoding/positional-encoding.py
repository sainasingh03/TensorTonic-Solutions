
import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):

    # Step 1: Create an empty positional encoding matrix
    pe = np.zeros((seq_len, d_model), dtype=float)

    # Step 2: Create position values: 0, 1, 2, ...
    positions = np.arange(seq_len)[:, np.newaxis]

    # Step 3: Calculate frequency divisors
    div_term = base ** (
        2 * np.arange((d_model + 1) // 2) / d_model
    )

    # Step 4: Calculate angles
    angles = positions / div_term

    # Step 5: Apply sine to even columns
    pe[:, 0::2] = np.sin(angles)

    # Step 6: Apply cosine to odd columns
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    # Step 7: Return the result
    return pe
