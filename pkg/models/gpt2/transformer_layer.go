package gpt2

import (
	"gonum.org/v1/gonum/mat"
)

type TransformerLayer struct {
    SelfAttention *SelfAttention
    FeedForward   *PositionWiseFeedForward
    LayerNorm1    *LayerNorm
    LayerNorm2    *LayerNorm
}

func NewTransformerLayer(hiddenSize, numHeads int) *TransformerLayer {
    selfAttention := NewSelfAttention(hiddenSize, numHeads)
    feedForward := NewPositionWiseFeedForward(hiddenSize)
    layerNorm1 := NewLayerNorm(hiddenSize)
    layerNorm2 := NewLayerNorm(hiddenSize)

    return &TransformerLayer{
        SelfAttention: selfAttention,
        FeedForward:   feedForward,
        LayerNorm1:    layerNorm1,
        LayerNorm2:    layerNorm2,
    }
}

func (t *TransformerLayer) SetWeights(weights map[string]*mat.Dense, layerKey string) {
    t.SelfAttention.SetWeights(weights, layerKey+".attn")
    t.FeedForward.SetWeights(weights, layerKey+".mlp")
    t.LayerNorm1.SetWeights(weights[layerKey+".ln_1.weight"], weights[layerKey+".ln_1.bias"])
    t.LayerNorm2.SetWeights(weights[layerKey+".ln_2.weight"], weights[layerKey+".ln_2.bias"])
}

func (tl *TransformerLayer) Forward(input *mat.Dense, mask *mat.Dense) *mat.Dense {
	// Multi-head self-attention
	attended := tl.mha.SelfAttention(input, mask)

	// Add & norm (residual connection)
	attendedNorm := tl.attentionNorm.AddAndNorm(input, attended)

	// Position-wise feed-forward
	ffnOut := tl.ffn.Forward(attendedNorm)

	// Add & norm (residual connection)
	output := tl.feedForwardNorm.AddAndNorm(attendedNorm, ffnOut)

	return output
}
