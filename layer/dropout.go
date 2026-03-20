package layer

import (
	"fmt"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type Dropout struct {
	P          float64
	IsTraining bool
	mask       *mat.Dense
}

func NewDropout(p float64) *Dropout {
	return &Dropout{
		P: p,
	}
}

func (l *Dropout) SetTraining(isTraining bool) {
	l.IsTraining = isTraining
}

func (l *Dropout) Forward(inputs *mat.Dense) *mat.Dense {
	if !l.IsTraining {
		return inputs
	}

	r, c := inputs.Dims()
	if l.mask == nil {
		l.mask = mat.NewDense(r, c, nil)
	}
	oldR, oldC := l.mask.Dims()
	if oldR != r || oldC != c {
		l.mask = mat.NewDense(r, c, nil)
	}

	scale := 1.0 / (1.0 - l.P)
	l.mask.Apply(func(i, j int, v float64) float64 {
		if l.P > rand.Float64() {
			return scale
		}
		return 0
	}, l.mask)

	var out mat.Dense
	out.MulElem(inputs, l.mask)
	return &out
}

func (l *Dropout) Backward(gradOutput *mat.Dense, lr float64) *mat.Dense {
	if l.mask == nil {
		return gradOutput
	}

	var downstreamGradient mat.Dense
	downstreamGradient.MulElem(gradOutput, l.mask)
	return &downstreamGradient
}

func (l *Dropout) String() string {
	return fmt.Sprintf("Dropout(p = %.2f)", l.P)
}
