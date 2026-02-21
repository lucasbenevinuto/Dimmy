import numpy as np
from numpy.testing import assert_array_almost_equal
from attention import scaled_dot_product_attention


Q = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
], dtype=np.float64)

K = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
], dtype=np.float64)

V = np.array([
    [1.0, 0.0, 2.0, 1.0],
    [0.0, 1.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 0.0],
], dtype=np.float64)


def compute_expected_output():
    d_k = K.shape[-1]
    scores = Q @ K.T
    scaled_scores = scores / np.sqrt(d_k)
    shifted = scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    expected_weights = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
    expected_output = expected_weights @ V
    return expected_output, expected_weights


def test_weights_sum_to_one(attention_weights):
    row_sums = np.sum(attention_weights, axis=-1)
    try:
        assert np.allclose(row_sums, 1.0, atol=1e-6)
        print("[PASSED] Cada linha dos attention_weights soma 1.0")
    except AssertionError:
        print("[FAILED] Cada linha dos attention_weights soma 1.0")
        print(f"  Somas obtidas: {row_sums}")


def test_output_shape(output):
    expected_shape = (Q.shape[0], V.shape[1])
    try:
        assert output.shape == expected_shape
        print(f"[PASSED] Shape do output é {expected_shape}")
    except AssertionError:
        print(f"[FAILED] Shape do output: esperado {expected_shape}, obtido {output.shape}")


def test_numerical_correctness(output, attention_weights):
    expected_output, expected_weights = compute_expected_output()
    try:
        assert_array_almost_equal(attention_weights, expected_weights)
        assert_array_almost_equal(output, expected_output)
        print("[PASSED] Valores numéricos conferem com cálculo manual")
    except AssertionError as e:
        print(f"[FAILED] Valores numéricos divergem: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("INPUTS")
    print("=" * 50)
    print(f"Q (shape {Q.shape}):\n{Q}\n")
    print(f"K (shape {K.shape}):\n{K}\n")
    print(f"V (shape {V.shape}):\n{V}\n")

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print("=" * 50)
    print("RESULTADOS")
    print("=" * 50)
    print(f"Attention Weights (shape {attention_weights.shape}):\n{attention_weights}\n")
    print(f"Output (shape {output.shape}):\n{output}\n")

    print("=" * 50)
    print("TESTES")
    print("=" * 50)
    test_weights_sum_to_one(attention_weights)
    test_output_shape(output)
    test_numerical_correctness(output, attention_weights)
