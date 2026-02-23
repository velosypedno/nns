package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SoftMaxCrossEntropy struct{}

func NewSoftMaxCrossEntropyFunc() *SoftMaxCrossEntropy {
	return &SoftMaxCrossEntropy{}
}

func (f *SoftMaxCrossEntropy) Transform(output *mat.Dense) *mat.Dense {
	r, c := output.Dims()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		row := output.RawRowView(i)
		resRow := result.RawRowView(i)

		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}

		sumExp := 0.0
		for j, v := range row {
			resRow[j] = math.Exp(v - maxVal)
			sumExp += resRow[j]
		}

		for j := range resRow {
			resRow[j] /= sumExp
		}
	}
	return result
}

func (f *SoftMaxCrossEntropy) Calculate(output, target *mat.Dense) float64 {
	probs := f.Transform(output)

	r, c := probs.Dims()
	var totalLoss float64
	const epsilon = 1e-15

	for i := 0; i < r; i++ {
		pRow := probs.RawRowView(i)
		tRow := target.RawRowView(i)

		for j := 0; j < c; j++ {
			if tRow[j] > 0 {
				totalLoss -= math.Log(pRow[j] + epsilon)
			}
		}
	}

	return totalLoss / float64(r)
}

func (f *SoftMaxCrossEntropy) Derivative(output, target *mat.Dense) *mat.Dense {
	probs := f.Transform(output)
	r, c := probs.Dims()

	result := mat.NewDense(r, c, nil)
	result.Sub(probs, target)
	return result
}
