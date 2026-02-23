package loss

import "gonum.org/v1/gonum/mat"

type MSE struct{}

func NewMSE() *MSE {
	return &MSE{}
}

func (*MSE) Calculate(output, target *mat.Dense) float64 {
	var diff mat.Dense
	diff.Sub(output, target)

	r, c := diff.Dims()
	sum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := diff.At(i, j)
			sum += v * v
		}
	}
	return sum / float64(r*c)
}

func (*MSE) Derivative(output, target *mat.Dense) *mat.Dense {
	r, c := output.Dims()
	currentGradient := mat.NewDense(r, c, nil)
	currentGradient.Sub(output, target)
	return currentGradient
}

func (*MSE) Transform(output *mat.Dense) *mat.Dense {
	return output
}
