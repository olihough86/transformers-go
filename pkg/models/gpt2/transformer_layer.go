package gpt2

import (
	"gonum.org/v1/gonum/mat"
)

type TransformerLayer struct {
	mha             *MultiHeadAttention
	attentionNorm   *LayerNorm
	ffn             *PositionWiseFeedForward
	feedForwardNorm *LayerNorm
}

func NewTransformerLayer(hiddenSize, nHead int) *TransformerLayer {
	return &TransformerLayer{
		mha:             NewMultiHeadAttention(hiddenSize, nHead),
		attentionNorm:   NewLayerNorm(hiddenSize),
		ffn:             NewPositionWiseFeedForward(hiddenSize),
		feedForwardNorm: NewLayerNorm(hiddenSize),
	}
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
