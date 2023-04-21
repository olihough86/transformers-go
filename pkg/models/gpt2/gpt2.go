package gpt2

import (
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"fmt"
)

type GPT2Model struct {
    Config           *GPT2Config
    TransformerLayers []*TransformerLayer
    EmbeddingLayer    *EmbeddingLayer
    LayerNorm         *LayerNorm
}

func (m *GPT2Model) SetLayerNormWeights(weight, bias *mat.Dense) {
    m.LayerNorm.SetWeights(weight, bias)
}

func NewGPT2Model(config *GPT2Config) *GPT2Model {
    transformerLayers := make([]*TransformerLayer, config.NLayers)
    for i := 0; i < config.NLayers; i++ {
        transformerLayers[i] = NewTransformerLayer(config.HiddenSize, config.NHead)
    }

    embeddingLayer := NewEmbeddingLayer(config.VocabSize, config.HiddenSize)
    layerNorm := NewLayerNorm(config.HiddenSize)

    return &GPT2Model{
        Config:           config,
        TransformerLayers: transformerLayers,
        EmbeddingLayer:    embeddingLayer,
        LayerNorm:         layerNorm,
    }
}

func (model *GPT2Model) LoadWeights(path string) error {
	_, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	// Parse the weights and load them into the model architecture
	// You'll need to implement the actual weight loading here
	return nil
}

func (model *GPT2Model) Forward(input *mat.Dense, mask *mat.Dense) *mat.Dense {
	// Convert input *mat.Dense to []int
    inputRows, inputCols := input.Dims()
    inputInts := make([]int, inputRows*inputCols)
    for i := 0; i < inputRows; i++ {
        for j := 0; j < inputCols; j++ {
            inputInts[i*inputCols+j] = int(input.At(i, j))
        }
    }

	// Apply the input embeddings
	embeddedInput := model.EmbeddingLayer.Embed(inputInts)

	// Pass the input through each transformer layer
	transformerOutput := embeddedInput
	for _, layer := range model.TransformerLayers {
		transformerOutput = layer.Forward(transformerOutput, mask)
	}

	// Return the final output
	return transformerOutput
}

func (m *GPT2Model) SetWeights(weights map[string]*mat.Dense) error {
    // Set the weights for the embedding layer
    m.EmbeddingLayer.SetWeights(weights)

    // Set the weights for each transformer layer
    for i := 0; i < len(m.TransformerLayers); i++ {
        layerKey := fmt.Sprintf("h.%d", i)
        m.TransformerLayers[i].SetWeights(weights, layerKey)
    }

    // Set the weights for the layer norm
    m.SetLayerNormWeights(weights["ln_f.weight"], weights["ln_f.bias"])
    return nil
}