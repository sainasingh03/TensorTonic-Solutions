
import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):

    # Step 1: Handle empty input
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    # Step 2: Find maximum length if not given
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    # Step 3: Create an array filled with pad_value
    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    # Step 4: Copy values from original sequences
    for i, seq in enumerate(seqs):

        # Keep only max_len elements
        length = min(len(seq), max_len)

        # Copy elements into the result
        result[i, :length] = seq[:length]

    # Step 5: Return the padded array
    return result
