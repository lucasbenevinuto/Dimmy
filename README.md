# Scaled Dot-Product Attention — LAB P1-01

## Descrição

Este repositório contém uma implementação do mecanismo de **Scaled Dot-Product Attention**, introduzido no paper *"Attention Is All You Need"* (Vaswani et al., 2017). O mecanismo de atenção é o bloco fundamental da arquitetura Transformer, que revolucionou o processamento de linguagem natural e diversas outras áreas de aprendizado de máquina.

A ideia central do self-attention é permitir que cada posição de uma sequência "preste atenção" a todas as outras posições, calculando um peso de relevância para cada par. Para isso, cada token da entrada é projetado em três representações: **Query (Q)**, **Key (K)** e **Value (V)**. A similaridade entre queries e keys determina quanto cada value contribui para a saída.

O cálculo é feito por produto escalar entre Q e K, seguido de uma normalização por softmax que transforma os scores em pesos de atenção (valores entre 0 e 1 que somam 1 por linha). Esses pesos são então usados para computar uma média ponderada dos values, produzindo a saída final do mecanismo.

## Como rodar

### Pré-requisitos

- Python 3.x
- NumPy

### Comandos

```bash
pip install numpy
python test_attention.py
```

## Explicação do Scaling Factor (√dₖ)

Quando a dimensionalidade das keys (dₖ) é grande, os produtos escalares entre queries e keys tendem a crescer em magnitude. Isso acontece porque o produto escalar de dois vetores de dimensão dₖ é a soma de dₖ termos, e sua variância cresce proporcionalmente a dₖ.

Valores muito grandes nos scores empurram a função softmax para regiões de **saturação**, onde os gradientes se tornam extremamente pequenos. Na prática, isso significa que o softmax produz distribuições quase one-hot, dificultando o aprendizado durante o treinamento por backpropagation.

Dividir os scores por **√dₖ** normaliza a variância dos produtos escalares de volta para aproximadamente 1, mantendo o softmax em uma região com gradientes mais saudáveis e permitindo uma distribuição de atenção mais suave e treinável.

## Exemplo de Input/Output

### Matrizes de entrada (3×4)

```
Q = [[1.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0, 0.0]]

K = [[1.0, 0.0, 0.0, 1.0],
     [0.0, 1.0, 1.0, 0.0],
     [1.0, 1.0, 0.0, 0.0]]

V = [[1.0, 0.0, 2.0, 1.0],
     [0.0, 1.0, 0.0, 2.0],
     [1.0, 1.0, 1.0, 0.0]]
```

### Attention Weights (3×3)

```
[[0.33333333, 0.33333333, 0.33333333],
 [0.33333333, 0.33333333, 0.33333333],
 [0.27406862, 0.27406862, 0.45186276]]
```

### Output (3×4)

```
[[0.66666667, 0.66666667, 1.00000000, 1.00000000],
 [0.66666667, 0.66666667, 1.00000000, 1.00000000],
 [0.72593138, 0.72593138, 1.00000000, 0.82220586]]
```

## Fórmula de Referência

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

## Referência

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. *Advances in Neural Information Processing Systems (NeurIPS)*.
