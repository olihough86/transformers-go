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
	//batchSize := 1

	// Create a TransformerLayer instance
	tLayer := gpt2.NewTransformerLayer(hiddenSize, nHead)

	// Generate a random input matrix
	input := mat.NewDense(inputLength, hiddenSize, randomArray(inputLength*hiddenSize, 0.0, 0.01))

	// Generate a random mask matrix
	qRows, _ := input.Dims()
	kRows, _ := input.Dims()
	mask := randomBinaryMask(qRows, kRows)

	// Perform a forward pass through the TransformerLayer
	output := tLayer.Forward(input, mask)

	// Print the output matrix
	fmt.Println("Output matrix:")
	fmt.Println(mat.Formatted(output))
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

