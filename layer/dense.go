package layer

import (
	"fmt"
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type Dense struct {
	Weights *mat.Dense
	Biases  *mat.Dense

	LastInputs *mat.Dense
}

func (l *Dense) String() string {
	r, c := l.Weights.Dims()
	weightsStr := fmt.Sprintf("%v", mat.Formatted(l.Weights, mat.Prefix("    "), mat.Squeeze()))
	biasesStr := fmt.Sprintf("%v", mat.Formatted(l.Biases, mat.Prefix("    "), mat.Squeeze()))

	return fmt.Sprintf(
		"Layer [%d -> %d]:\n  Weights:\n%s\n  Biases:\n%s",
		r, c, weightsStr, biasesStr,
	)
}

func randomInit(r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.NormFloat64() * math.Sqrt(1.0/float64(r))
	}
	return mat.NewDense(r, c, data)
}

func NewDense(r, c int) *Dense {
	weights := randomInit(r, c)
	biases := mat.NewDense(1, c, nil)
	return &Dense{
		Weights: weights,
		Biases:  biases,

		LastInputs: new(mat.Dense),
	}
}

func (l *Dense) Forward(inputs *mat.Dense) *mat.Dense {
	l.LastInputs = mat.DenseCopyOf(inputs)

	var out mat.Dense
	out.Mul(inputs, l.Weights)

	r, _ := inputs.Dims()
	for i := 0; i < r; i++ {
		row := out.RawRowView(i)
		for j, b := range l.Biases.RawRowView(0) {
			row[j] += b
		}
	}

	return &out
}

func (l *Dense) Backward(upstreamGradient *mat.Dense, lr float64) *mat.Dense {
	var currentGrad mat.Dense
	currentGrad.Mul(l.LastInputs.T(), upstreamGradient)

	var downstreamGradient mat.Dense
	downstreamGradient.Mul(upstreamGradient, l.Weights.T())

	currentGrad.Scale(lr, &currentGrad)

	l.Weights.Sub(l.Weights, &currentGrad)

	rows, cols := upstreamGradient.Dims()

	// correct biases
	gradSum := make([]float64, cols)
	for i := 0; i < rows; i++ {
		row := upstreamGradient.RawRowView(i)
		for j, val := range row {
			gradSum[j] += val
		}
	}
	biasRow := l.Biases.RawRowView(0)
	for j := range biasRow {
		biasRow[j] -= (gradSum[j] / float64(rows) * lr)
	}

	return &downstreamGradient
}
