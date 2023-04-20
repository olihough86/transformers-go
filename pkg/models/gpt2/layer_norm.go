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
		gain: mat.NewDense(1, hiddenSize, randomArray(hiddenSize, 0.0, 0.01)),
		bias: mat.NewDense(1, hiddenSize, randomArray(hiddenSize, 0.0, 0.01)),
		eps:  1e-5,
	}
}

func (ln *LayerNorm) AddAndNorm(a, b *mat.Dense) *mat.Dense {
	sum := mat.Dense{}
	sum.Add(a, b)
	mean, variance := meanAndVariance(&sum)
	norm := mat.Dense{}
	norm.Apply(func(i, j int, v float64) float64 {
		return v - mean.At(i, 0)
	}, &sum)


	// Compute the square root element-wise
	sqrtVariance := mat.Dense{}
	sqrtVariance.Apply(func(_, _ int, v float64) float64 {
		return math.Sqrt(v)
	}, variance)

	// Add the epsilon term
	sqrtVariance.Apply(func(_, _ int, v float64) float64 {
		return v + ln.eps
	}, &sqrtVariance)

	norm.Apply(func(i, j int, v float64) float64 {
		return v / sqrtVariance.At(i, 0)
	}, &norm)
	norm.Apply(func(i, j int, v float64) float64 {
		return v * ln.gain.At(0, j)
	}, &norm)
	norm.Apply(func(i, j int, v float64) float64 {
		return v + ln.bias.At(0, j)
	}, &norm)
 	norm.Apply(func(i, j int, v float64) float64 {
		return v + ln.bias.At(0, j)
	}, &norm)
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

