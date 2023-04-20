package gpt2

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type MultiHeadAttention struct {
	queryProjection *mat.Dense
	keyProjection   *mat.Dense
	valueProjection *mat.Dense
	outProjection   *mat.Dense
	nHead           int
	dHead           int
}

func scaledDotProductAttention(q, k, v, mask *mat.Dense, dHead int) *mat.Dense {
	// Calculate Q * K^T
	kt := mat.DenseCopyOf(k.T())
	qk := mat.Dense{}
	qk.Mul(q, kt)

	// Scale the dot products
	scale := 1.0 / math.Sqrt(float64(dHead))
	qk.Scale(scale, &qk)

	// Apply the mask if provided
	if mask != nil {
		qk.Apply(func(i, j int, v float64) float64 {
			return v + mask.At(i, j)
		}, &qk)
	}

	// Softmax
	qk.Apply(func(_, _ int, v float64) float64 {
		return math.Exp(v)
	}, &qk)

	rows, cols := qk.Dims()
	sum := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		rowSum := 0.0
		for j := 0; j < cols; j++ {
			rowSum += qk.At(i, j)
		}
		for j := 0; j < cols; j++ {
			sum.Set(i, j, rowSum)
		}
	}

	qk.DivElem(&qk, sum)

	// Calculate the attended values: qk * V
	attended := mat.Dense{}
	attended.Mul(&qk, v)

	return &attended
}

func NewMultiHeadAttention(hiddenSize, nHead int) *MultiHeadAttention {
	dHead := hiddenSize / nHead
	return &MultiHeadAttention{
		nHead: nHead,
		dHead: dHead,
		// Initialize the projection matrices with random values.
		// You'll replace these with the actual pre-trained weights later.
		queryProjection: mat.NewDense(hiddenSize, hiddenSize, randomArray(hiddenSize*hiddenSize)),
		keyProjection:   mat.NewDense(hiddenSize, hiddenSize, randomArray(hiddenSize*hiddenSize)),
		valueProjection: mat.NewDense(hiddenSize, hiddenSize, randomArray(hiddenSize*hiddenSize)),
		outProjection:   mat.NewDense(hiddenSize, hiddenSize, randomArray(hiddenSize*hiddenSize)),
	}
}

func randomArray(size int) []float64 {
	randArray := make([]float64, size)
	for i := range randArray {
		randArray[i] = rand.Float64() // Use a random number generator of your choice
	}
	return randArray
}

func (mha *MultiHeadAttention) SelfAttention(input *mat.Dense, mask *mat.Dense) *mat.Dense {
	// Check the dimensions of the input and projection matrices
	inputRows, inputCols := input.Dims()
	fmt.Printf("input dims: %d x %d\n", inputRows, inputCols)

	queryProjectionRows, queryProjectionCols := mha.queryProjection.Dims()
	fmt.Printf("queryProjection dims: %d x %d\n", queryProjectionRows, queryProjectionCols)

	// Project the input to query, key, and value metrics
	query := mat.Dense{}
	query.Mul(input, mha.queryProjection)

	key := mat.Dense{}
	key.Mul(input, mha.keyProjection)

	value := mat.Dense{}
	value.Mul(input, mha.valueProjection)

	// Split the projected matrices into multiple heads
	queryHeads, _ := splitHeads(&query, mha.nHead)
	keyHeads, _ := splitHeads(&key, mha.nHead)
	valueHeads, _ := splitHeads(&value, mha.nHead)

	// Compute the self-attention for each head
	attendedHeads := make([]*mat.Dense, mha.nHead)
	for i := 0; i < mha.nHead; i++ {
		attendedHeads[i] = scaledDotProductAttention(queryHeads[i], keyHeads[i], valueHeads[i], mask, mha.dHead)
	}

	// Concatenate the attended heads
	attended := concatHeads(attendedHeads, mha.nHead)

	// Project the concatenated matrix to the output size
	output := mat.Dense{}
	output.Mul(attended, mha.outProjection)

	return &output
}

func splitHeads(matrix *mat.Dense, nHead int) ([]*mat.Dense, int) {
    rows, cols := matrix.Dims()
    headSize := cols / nHead
    heads := make([]*mat.Dense, nHead)
    for i := 0; i < nHead; i++ {
        headData := make([]float64, rows*headSize)
        for j := 0; j < rows; j++ {
            for k := 0; k < headSize; k++ {
                headData[j*headSize+k] = matrix.At(j, i*headSize+k)
            }
        }
        heads[i] = mat.NewDense(rows, headSize, headData)
    }
    return heads, headSize
}

func concatHeads(heads []*mat.Dense, nHead int) *mat.Dense {
	rows, headSize := heads[0].Dims()
	concat := mat.NewDense(rows, headSize*nHead, nil)
	for i, head := range heads {
		for j := 0; j < rows; j++ {
			for k := 0; k < headSize; k++ {
				concat.Set(j, i*headSize+k, head.At(j, k))
			}
		}
	}
	return concat
}

