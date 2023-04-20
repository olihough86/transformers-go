package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/olihough86/transformers-go/pkg/models/gpt2"
	"github.com/olihough86/transformers-go/utils"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Load the GPT2 config
	config := gpt2.NewGPT2Config()

	// Create a new GPT2 model
	model := gpt2.NewGPT2Model(config)

	// Load the model weights
	weightsFile := "/home/nisnet/gpt2/pytorch_model.bin"
	weights, err := utils.LoadWeights(weightsFile)
	if err != nil {
		fmt.Println("Error loading weights:", err)
		return
	}

	// Set the weights in the model
	model.SetWeights(weights)

	// Your code to use the GPT2 model
}

func randomArray(size int, mean float64, stddev float64) []float64 {
	randArray := make([]float64, size)
	for i := range randArray {
		randArray[i] = rand.NormFloat64()*stddev + mean // Use a random number generator of your choice
	}
	return randArray
}

func randomBinaryMask(rows, cols int) *mat.Dense {
	mask := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			mask.Set(i, j, float64(rand.Intn(2))) // Generates either 0 or 1
		}
	}
	return mask
}
