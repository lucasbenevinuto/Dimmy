import numpy as np


def softmax(x):
    """Aplica softmax linha a linha em uma matriz 2D.

    Utiliza o truque de estabilidade numérica, subtraindo o valor
    máximo de cada linha antes de exponenciar, evitando overflow.

    Args:
        x: Matriz 2D (shape: [linhas, colunas]).

    Returns:
        Matriz com mesma shape, onde cada linha soma 1.
    """
    shifted_values = x - np.max(x, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=-1, keepdims=True)
    return exponentials / row_sums
