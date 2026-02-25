package layer

import (
	"gonum.org/v1/gonum/mat"
)

type ReLU struct {
	lastInputs *mat.Dense
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (l *ReLU) Forward(inputs *mat.Dense) *mat.Dense {
	l.lastInputs = mat.DenseCopyOf(inputs)

	r, c := inputs.Dims()
	data := inputs.RawMatrix().Data
	outputData := make([]float64, len(data))

	for i, val := range data {
		if val > 0 {
			outputData[i] = val
		} else {
			outputData[i] = 0
		}
	}

	return mat.NewDense(r, c, outputData)
}

func (l *ReLU) Backward(gradOutput *mat.Dense, lr float64) *mat.Dense {
	r, c := gradOutput.Dims()

	gradData := gradOutput.RawMatrix().Data
	inputData := l.lastInputs.RawMatrix().Data

	dXData := make([]float64, len(gradData))

	for i := range gradData {
		if inputData[i] > 0 {
			dXData[i] = gradData[i]
		} else {
			dXData[i] = 0
		}
	}

	return mat.NewDense(r, c, dXData)
}

func (l *ReLU) GobEncode() ([]byte, error) {
	return []byte{}, nil
}

func (l *ReLU) GobDecode(data []byte) error {
	return nil
}
