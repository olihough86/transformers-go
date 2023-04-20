package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/olihough86/transformers-go/pkg/models/gpt2"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Set the random seed for reproducibility
	rand.Seed(time.Now().UnixNano())

	// Set some example hyperparameters
	hiddenSize := 768
	nHead := 12
	inputLength := 10
	batchSize := 1

	// Create a TransformerLayer instance
	tLayer := gpt2.NewTransformerLayer(hiddenSize, nHead)

	// Generate a random input matrix
	input := mat.NewDense(batchSize, inputLength*hiddenSize, randomArray(batchSize*inputLength*hiddenSize))

	// Generate a random mask matrix
	mask := mat.NewDense(batchSize, inputLength, randomArray(batchSize*inputLength))

	// Perform a forward pass through the TransformerLayer
	output := tLayer.Forward(input, mask)

	// Print the output matrix
	fmt.Println("Output matrix:")
	fmt.Println(mat.Formatted(output))
}

func randomArray(size int) []float64 {
	randArray := make([]float64, size)
	for i := range randArray {
		randArray[i] = rand.Float64() // Use a random number generator of your choice
	}
	return randArray
}
