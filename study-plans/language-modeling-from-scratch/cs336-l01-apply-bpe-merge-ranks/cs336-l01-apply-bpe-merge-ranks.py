
def encode(text, merges):
    tokens = list(text.encode("utf-8"))

    for left, right, new_id in merges:
        result = []
        i = 0

        while i < len(tokens):
            if (i + 1 < len(tokens) and
                tokens[i] == left and
                tokens[i + 1] == right):

                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        tokens = result

    return tokens


def decode(ids, vocab):
    result = bytearray()

    for token_id in ids:
        result.extend(vocab[token_id])

    return result.decode("utf-8")
