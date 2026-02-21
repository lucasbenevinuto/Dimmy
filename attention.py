from typing import Tuple
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:

    if x.ndim != 2:
        raise ValueError(
            f"softmax espera uma matriz 2D, mas recebeu array com {x.ndim} dimensão(ões)."
        )

    shifted_values = x - np.max(x, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=-1, keepdims=True)
    return exponentials / row_sums


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    for name, array in [("Q", Q), ("K", K), ("V", V)]:
        if array.ndim != 2:
            raise ValueError(
                f"{name} deve ser uma matriz 2D, mas tem {array.ndim} dimensão(ões)."
            )

    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"Dimensão dₖ incompatível: Q.shape[1]={Q.shape[1]} != K.shape[1]={K.shape[1]}."
        )

    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"Número de linhas incompatível: K.shape[0]={K.shape[0]} != V.shape[0]={V.shape[0]}."
        )

    scaling_factor = np.sqrt(K.shape[-1])
    scores = Q @ K.T
    scaled_scores = scores / scaling_factor
    attention_weights = softmax(scaled_scores)
    output = attention_weights @ V
    return output, attention_weights
