package gpt2

import (
    "gonum.org/v1/gonum/mat"
)

type TransformerLayer struct {
    MultiHeadAttention *MultiHeadAttention
    FeedForward        *PositionWiseFeedForward
    LayerNorm1         *LayerNorm
    LayerNorm2         *LayerNorm
}

func NewTransformerLayer(hiddenSize, numHeads int) *TransformerLayer {
    mha := NewMultiHeadAttention(hiddenSize, numHeads)
    feedForward := NewPositionWiseFeedForward(hiddenSize)
    layerNorm1 := NewLayerNorm(hiddenSize)
    layerNorm2 := NewLayerNorm(hiddenSize)

    return &TransformerLayer{
        MultiHeadAttention: mha,
        FeedForward:        feedForward,
        LayerNorm1:         layerNorm1,
        LayerNorm2:         layerNorm2,
    }
}

func (tl *TransformerLayer) SetWeights(weights map[string]*mat.Dense, layerKey string) {
    tl.MultiHeadAttention.SetWeights(weights, layerKey+".attn")
    // Implement SetWeights for PositionWiseFeedForward and call it here
    // tl.FeedForward.SetWeights(weights, layerKey+".mlp")
    tl.LayerNorm1.SetWeights(weights[layerKey+".ln_1.weight"], weights[layerKey+".ln_1.bias"])
    tl.LayerNorm2.SetWeights(weights[layerKey+".ln_2.weight"], weights[layerKey+".ln_2.bias"])
}

func (tl *TransformerLayer) Forward(input *mat.Dense, mask *mat.Dense) *mat.Dense {
    // Multi-head self-attention
    attended := tl.MultiHeadAttention.SelfAttention(input, mask)

    // Add & norm (residual connection)
    attendedNorm := tl.LayerNorm1.AddAndNorm(input, attended)

    // Position-wise feed-forward
    ffnOut := tl.FeedForward.Forward(attendedNorm)

    // Add & norm (residual connection)
    output := tl.LayerNorm2.AddAndNorm(attendedNorm, ffnOut)

    return output
}