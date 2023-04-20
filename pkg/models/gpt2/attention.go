package gpt2

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type MultiHeadAttention struct {
	nHead          int
	queryProjection *mat.Dense
	keyProjection   *mat.Dense
	valueProjection *mat.Dense
	outProjection   *mat.Dense
}

func scaledDotProductAttention(query, key, value *mat.Dense, mask *mat.Dense) *mat.Dense {
	// Calculate Q * K^T
	kt := mat.DenseCopyOf(key.T())
	qk := mat.Dense{}
	qk.Mul(query, kt)

	// Scale the dot products
	scale := 1.0 / math.Sqrt(float64(query.RawMatrix().Cols))
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

	sum := mat.NewVecDense(qk.RawMatrix().Rows, nil)
	for i := 0; i < qk.RawMatrix().Rows; i++ {
		rowSum := 0.0
		for j := 0; j < qk.RawMatrix().Cols; j++ {
			rowSum += qk.At(i, j)
		}
		sum.SetVec(i, rowSum)
	}
	qk.DivElem(&qk, sum)

	// Calculate the attended values: qk * V
	attended := mat.Dense{}
	attended.Mul(&qk, value)

	return &attended
}

func NewMultiHeadAttention(hiddenSize, nHead int) *MultiHeadAttention {
	return &MultiHeadAttention{
		nHead: nHead,
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
	// Project the input to query, key, and value metrics
	query := mat.Dense{}
	query.Mul(input, mha.queryProjection)

	key := mat.Dense{}
	key.Mul(input, mha.keyProjection)

	value := mat.Dense{}
	value.Mul(input, mha.valueProjection)

	// Split the projected matrices into multiple heads
	queryHeads := splitHeads(&query, mha.nHead)
	keyHeads := splitHeads(&key, mha.nHead)
	valueHeads := splitHeads(&value, mha.nHead)

	// Compute the self-attention for each head
	attendedHeads := make([]*mat.Dense, mha.nHead)
	for i := 0; i < mha.nHead; i++ {
		attendedHeads[i] = scaledDotProductAttention(queryHeads[i], keyHeads[i], valueHeads[i], mask)
	}

	// Concatenate the attended heads
	attended := concatHeads(attendedHeads)

	// Project the concatenated matrix to the output size
	output := mat.Dense{}
	output.Mul(attended, mha.outProjection)

	return &output
}

func splitHeads(matrix *mat.Dense, nHead int) []*mat.Dense {
	rows, cols := matrix.Dims()
	headSize := cols / nHead
	heads := make([]*mat.Dense, nHead)
	for i := 0; i < nHead; i++ {
		headData := matrix.RawMatrix().Data[i*headSize : (i+1)*headSize]
		heads[i] = mat.NewDense(rows, headSize, headData)
	}
	return heads
}

func concatHeads(heads []*mat.Dense) *mat.Dense {
	rows, headSize := heads[0].Dims()
	nHead := len(heads)
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