package gpt2

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type LayerNorm struct {
	gain *mat.Dense
	bias *mat.Dense
	eps  float64
}

func NewLayerNorm(hiddenSize int) *LayerNorm {
	return &LayerNorm{
		gain: mat.NewDense(1, hiddenSize, randomArray(hiddenSize)),
		bias: mat.NewDense(1, hiddenSize, randomArray(hiddenSize)),
		eps:  1e-5,
	}
}

func (ln *LayerNorm) AddAndNorm(a, b *mat.Dense) *mat.Dense {
	sum := mat.Dense{}
	sum.Add(a, b)
	mean, variance := meanAndVariance(&sum)
	norm := mat.Dense{}
	norm.Sub(&sum, mean)

	// Compute the square root element-wise
	sqrtVariance := mat.Dense{}
	sqrtVariance.Apply(func(_, _ int, v float64) float64 {
		return math.Sqrt(v)
	}, variance)

	// Add the epsilon term
	sqrtVariance.Apply(func(_, _ int, v float64) float64 {
		return v + ln.eps
	}, &sqrtVariance)

	norm.DivElem(&norm, &sqrtVariance)
	norm.MulElem(&norm, ln.gain)
	norm.Add(&norm, ln.bias)
	return &norm
}


// Helper function to compute mean and variance
func meanAndVariance(matrix *mat.Dense) (mean, variance *mat.Dense) {
	rows, cols := matrix.Dims()
	mean = mat.NewDense(rows, 1, nil)
	variance = mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		rowMean := mat.Sum(matrix.RowView(i)) / float64(cols)
		mean.Set(i, 0, rowMean)

		rowVariance := 0.0
		for j := 0; j < cols; j++ {
			diff := matrix.At(i, j) - rowMean
			rowVariance += diff * diff
		}
		rowVariance /= float64(cols)
		variance.Set(i, 0, rowVariance)
	}

	return
}

