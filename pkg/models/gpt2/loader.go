package gpt2

import (
	"errors"
	"strings"

	"gonum.org/v1/gonum/mat"
	"github.com/wangkuiyi/gotorch"
)

func extractWeights(checkpoint map[string]*mat.Dense) (map[string]*mat.Dense, error) {
	weights := make(map[string]*mat.Dense)

	for key, value := range checkpoint {
		// Compute the dimensions of the weight matrix
		rows, cols := computeWeightDimensions(key)
		if rows == 0 || cols == 0 {
			return nil, errors.New("invalid dimensions for weight matrix: " + key)
		}

		// Assign the value directly to the weights map
		weights[key] = value
	}

	return weights, nil
}

func LoadGPT2Model(configFile string, checkpointFile string) (*GPT2Model, error) {
	// Load the GPT-2 model configuration
	config, err := LoadConfig(configFile)
	if err != nil {
		return nil, err
	}

	// Load the checkpoint file
	checkpoint, err := loadTorchCheckpoint(checkpointFile)
	if err != nil {
		return nil, err
	}

	// Extract the weights for each layer and component
	weights, err := extractWeights(checkpoint)
	if err != nil {
		return nil, err
	}

	// Create a new GPT-2 model instance
	model := NewGPT2Model(config)

	// Set the weights for each layer using the extracted weights
	err = model.SetWeights(weights)
	if err != nil {
		return nil, err
	}

	return model, nil
}

func computeWeightDimensions(key string) (int, int) {
    // Implement this function based on your model's architecture.
    // You should return the dimensions (rows, cols) based on the key.
    // This is just a placeholder example:
    if strings.HasPrefix(key, "wte") {
        return 768, 50257
    } else if strings.HasPrefix(key, "h") {
        if strings.Contains(key, "attn") {
            return 64, 768
        } else if strings.Contains(key, "mlp") {
            return 3072, 768
        } else if strings.Contains(key, "ln_") {
            return 1, 768
        }
    }
    return 0, 0
}

func loadTorchCheckpoint(checkpointFile string) (map[string]*mat.Dense, error) {
    // Load the PyTorch checkpoint
    checkpointTorch := gotorch.Load(checkpointFile)

    // Deserialize the state_dict
    stateDict := make(map[string]*gotorch.Tensor)
    err := checkpointTorch.To(stateDict)
    if err != nil {
        return nil, err
    }

    // Convert the gotorch.Tensor to a map[string]*mat.Dense
    checkpoint := make(map[string]*mat.Dense)
    for key, value := range stateDict {
        checkpoint[key] = torchTensorToMatDense(value)
    }

    return checkpoint, nil
}

func torchTensorToMatDense(tensor *gotorch.Tensor) *mat.Dense {
    shape := tensor.Shape()
    rows, cols := shape[0], shape[1]
    data := tensor.Totype(gotorch.Float64, true).([]float64)
    return mat.NewDense(rows, cols, data)
}