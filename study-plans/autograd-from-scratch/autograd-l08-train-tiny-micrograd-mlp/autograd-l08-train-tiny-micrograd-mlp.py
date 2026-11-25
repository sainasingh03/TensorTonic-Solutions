import torch

def train_tiny_micrograd_mlp(inputs, targets, weights, biases, learning_rate, steps):
    W = [w.clone() for w in weights]
    b = [bi.clone() for bi in biases]

    loss_history = []

    def forward():
        h = inputs
        activations = [h]

        for wi, bi in zip(W, b):
            h = torch.tanh(h @ wi.T + bi)
            activations.append(h)

        return h.squeeze(-1), activations

    for _ in range(steps):
        predictions, activations = forward()

        # Record loss BEFORE update
        loss = ((predictions - targets) ** 2).sum()
        loss_history.append(loss.clone())

        # Empty batch: no parameter updates
        if inputs.shape[0] == 0:
            continue

        # Manual backpropagation
        L = len(W)
        grads_W = [None] * L
        grads_b = [None] * L

        # Gradient at final layer
        delta = (2 * (predictions - targets).unsqueeze(1)
                 * (1 - activations[-1] ** 2))

        for layer in range(L - 1, -1, -1):
            prev_h = activations[layer]

            grads_W[layer] = delta.T @ prev_h
            grads_b[layer] = delta.sum(dim=0)

            if layer > 0:
                delta = (delta @ W[layer]) * (1 - activations[layer] ** 2)

        # Simultaneous update using gradients from old parameters
        W = [wi - learning_rate * gw for wi, gw in zip(W, grads_W)]
        b = [bi - learning_rate * gb for bi, gb in zip(b, grads_b)]

    # Final predictions and loss after all updates
    predictions, _ = forward()
    final_loss = ((predictions - targets) ** 2).sum()

    return predictions, final_loss, W, b, loss_history