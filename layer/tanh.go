package layer

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type Tanh struct {
	LastOutputs *mat.Dense
}

func NewTanh() *Tanh {
	return &Tanh{}
}

func (l *Tanh) String() string {
	if l.LastOutputs == nil {
		return "Activation: Tanh (not initialized)"
	}
	_, cols := l.LastOutputs.Dims()
	return fmt.Sprintf("Activation: Tanh (Features: %d)", cols)
}

func (l *Tanh) Forward(inputs *mat.Dense) *mat.Dense {
	rows, cols := inputs.Dims()
	out := mat.NewDense(rows, cols, nil)

	out.Apply(func(_, _ int, v float64) float64 {
		return math.Tanh(v)
	}, inputs)

	l.LastOutputs = out
	return out
}

func (l *Tanh) Backward(upstreamGradient *mat.Dense, lr float64) *mat.Dense {
	rows, cols := upstreamGradient.Dims()
	downstream := mat.NewDense(rows, cols, nil)

	downstream.Apply(func(r, c int, v float64) float64 {
		out := l.LastOutputs.At(r, c)
		derivative := 1 - (out * out)
		return v * derivative
	}, upstreamGradient)

	return downstream
}
