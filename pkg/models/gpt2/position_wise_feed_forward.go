package gpt2

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type PositionWiseFeedForward struct {
	w1 *mat.Dense
	w2 *mat.Dense
	b1 *mat.Dense
	b2 *mat.Dense
}

func NewPositionWiseFeedForward(hiddenSize int) *PositionWiseFeedForward {
	mean := 0.0
	stddev := 0.01

	return &PositionWiseFeedForward{
		w1: mat.NewDense(hiddenSize, hiddenSize, randomArray(hiddenSize*hiddenSize, mean, stddev)),
		w2: mat.NewDense(hiddenSize, hiddenSize, randomArray(hiddenSize*hiddenSize, mean, stddev)),
		b1: mat.NewDense(1, hiddenSize, randomArray(hiddenSize, mean, stddev)),
		b2: mat.NewDense(1, hiddenSize, randomArray(hiddenSize, mean, stddev)),
	}
}

func (pwff *PositionWiseFeedForward) Forward(input *mat.Dense) *mat.Dense {
	hidden := mat.Dense{}
	hidden.Mul(input, pwff.w1)
	hidden.Add(&hidden, pwff.b1)

	// Apply the activation function (e.g., ReLU or GeLU)
	hidden.Apply(activationFunction, &hidden)

	output := mat.Dense{}
	output.Mul(&hidden, pwff.w2)
	output.Add(&output, pwff.b2)

	return &output
}

func activationFunction(_, _ int, v float64) float64 {
	// Implement the activation function of your choice, e.g., ReLU or GeLU
	return v * (1.0 / (1.0 + math.Exp(-v))) // This is an example of the sigmoid function
}