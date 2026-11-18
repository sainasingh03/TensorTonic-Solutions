
def train_bpe(corpus, vocab_size):
    # Initial vocabulary: each byte maps to itself
    vocab_bytes = {
        i: bytes([i]) for i in range(256)
    }

    # Convert every string into UTF-8 byte token IDs
    sequences = [
        list(text.encode("utf-8"))
        for text in corpus
    ]

    vocab = []
    merges = []

    next_id = 256

    while next_id < vocab_size:

        # Count adjacent pairs across all strings
        pair_counts = {}

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + 1
                )

        # Stop if no adjacent pair exists
        if not pair_counts:
            break

        # Highest frequency wins.
        # On ties, compare actual byte strings.
        best_pair = max(
            pair_counts,
            key=lambda p: (
                pair_counts[p],
                vocab_bytes[p[0]],
                vocab_bytes[p[1]]
            )
        )

        left, right = best_pair

        # Create the new token by concatenating bytes
        new_token = next_id
        new_bytes = vocab_bytes[left] + vocab_bytes[right]

        vocab_bytes[new_token] = new_bytes

        vocab.append([
            new_token,
            list(new_bytes)
        ])

        merges.append([
            left,
            right,
            new_token
        ])

        # Replace matching pairs from left to right
        for seq_index, seq in enumerate(sequences):
            new_seq = []
            i = 0

            while i < len(seq):
                if (
                    i + 1 < len(seq)
                    and seq[i] == left
                    and seq[i + 1] == right
                ):
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            sequences[seq_index] = new_seq

        next_id += 1

    return {
        "vocab": vocab,
        "merges": merges
    }
