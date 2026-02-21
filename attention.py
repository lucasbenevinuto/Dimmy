import numpy as np


def softmax(x):

    shifted_values = x - np.max(x, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=-1, keepdims=True)
    return exponentials / row_sums


def scaled_dot_product_attention(Q, K, V):

    scaling_factor = np.sqrt(K.shape[-1])
    scores = Q @ K.T
    scaled_scores = scores / scaling_factor
    attention_weights = softmax(scaled_scores)
    output = attention_weights @ V
    return output, attention_weights
