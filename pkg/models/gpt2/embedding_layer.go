package gpt2

import (
	"gonum.org/v1/gonum/mat"
)

type EmbeddingLayer struct {
	weights *mat.Dense
}

func (e *EmbeddingLayer) SetWeights(weights map[string]*mat.Dense) {
    	e.weights = weights["wte"]
}
func (t *TransformerLayer) SetWeights(weights map[string]*mat.Dense, layerKey string) {
    	t.SelfAttention.SetWeights(weights, layerKey+".attn")
    	t.FeedForward.SetWeights(weights, layerKey+".mlp")
    	t.LayerNorm1.SetWeights(weights[layerKey+".ln_1.weight"], weights[layerKey+".ln_1.bias"])
    	t.LayerNorm2.SetWeights(weights[layerKey+".ln_2.weight"], weights[layerKey+".ln_2.bias"])
}

func NewEmbeddingLayer(vocabSize, embeddingSize int) *EmbeddingLayer {
	return &EmbeddingLayer{
		weights: mat.NewDense(vocabSize, embeddingSize, randomArray(vocabSize*embeddingSize, 0.0, 0.01)),
	}
}

func (el *EmbeddingLayer) Embed(inputTokens []int) *mat.Dense {
	tokenCount := len(inputTokens)
	_, embeddingSize := el.weights.Dims()

	embedding := mat.NewDense(tokenCount, embeddingSize, nil)

	for i, token := range inputTokens {
		tokenVector := el.weights.RawRowView(token)
		embedding.SetRow(i, tokenVector)
	}

	return embedding
}
