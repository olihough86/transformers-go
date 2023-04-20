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
    inputRows, inputCols := input.Dims()

    hidden := mat.NewDense(inputRows, inputCols, nil)
    hidden.Mul(input, pwff.w1)

    // Broadcast the addition of the bias matrix b1 across the rows of the hidden matrix
    b1Vec := pwff.b1.RowView(0).(*mat.VecDense) // Convert b1 to a *mat.VecDense
    for i := 0; i < inputRows; i++ {
        row := hidden.RowView(i).(*mat.VecDense) // Convert the row to a *mat.VecDense
        row.AddVec(row, b1Vec)
    }

    // Apply the activation function (e.g., ReLU or GeLU)
    hidden.Apply(activationFunction, hidden)

    output := mat.NewDense(inputRows, inputCols, nil)
    output.Mul(hidden, pwff.w2)

    // Broadcast the addition of the bias matrix b2 across the rows of the output matrix
    b2Vec := pwff.b2.RowView(0).(*mat.VecDense) // Convert b2 to a *mat.VecDense
    for i := 0; i < inputRows; i++ {
        row := output.RowView(i).(*mat.VecDense) // Convert the row to a *mat.VecDense
        row.AddVec(row, b2Vec)
    }

    return output
}



func activationFunction(_, _ int, v float64) float64 {
	// Implement the activation function of your choice, e.g., ReLU or GeLU
	return v * (1.0 / (1.0 + math.Exp(-v))) // This is an example of the sigmoid function
}