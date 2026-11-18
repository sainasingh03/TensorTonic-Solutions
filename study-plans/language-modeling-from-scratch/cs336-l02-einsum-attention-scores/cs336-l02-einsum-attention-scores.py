
import torch
import math

def attention_scores(q, k, num_heads):

    B, S_q, D = q.shape
    _, S_k, _ = k.shape

    d_h = D // num_heads

    # Split model dimension into attention heads
    q = q.reshape(B, S_q, num_heads, d_h)
    k = k.reshape(B, S_k, num_heads, d_h)

    # Move heads before sequence dimensions
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    # Calculate scaled dot-product attention
    scores = torch.einsum(
        'bhid,bhjd->bhij', q, k
    ) / math.sqrt(d_h)

    return scores
