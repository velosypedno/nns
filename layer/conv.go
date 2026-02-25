package layer

import (
	"github.com/velosypedno/nns/im2col"
	"gonum.org/v1/gonum/mat"
)

type Conv struct {
	KernelSize    int
	KernelsAmount int
	InChannels    int
	InR, InC      int

	Kernels *mat.Dense
	Biases  *mat.Dense

	lastIm2Col *mat.Dense
}

func NewConv(kernelSize, kernelsAmount, inChannels, inR, inC int) *Conv {
	weightRows := kernelsAmount
	weightCols := inChannels * kernelSize * kernelSize

	kernels := randomInit(weightRows, weightCols)
	biases := mat.NewDense(1, kernelsAmount, nil)

	return &Conv{
		KernelSize:    kernelSize,
		KernelsAmount: kernelsAmount,
		InChannels:    inChannels,
		InR:           inR,
		InC:           inC,
		Kernels:       kernels,
		Biases:        biases,
	}
}
func (l *Conv) Forward(inputs *mat.Dense) *mat.Dense {
	batchSize, _ := inputs.Dims()

	outR := l.InR - l.KernelSize + 1
	outC := l.InC - l.KernelSize + 1
	numWindows := outR * outC

	windows := im2col.ToWindowsMultiChannel(inputs, l.InChannels, l.InR, l.InC, l.KernelSize)
	l.lastIm2Col = windows

	var rawResult mat.Dense
	rawResult.Mul(l.Kernels, windows)

	data := make([]float64, batchSize*l.KernelsAmount*numWindows)
	for b := 0; b < batchSize; b++ {
		for k := 0; k < l.KernelsAmount; k++ {
			bias := l.Biases.At(0, k)
			for w := 0; w < numWindows; w++ {
				val := rawResult.At(k, b*numWindows+w)
				outIdx := b*(l.KernelsAmount*numWindows) + k*numWindows + w
				data[outIdx] = val + bias
			}
		}
	}

	return mat.NewDense(batchSize, l.KernelsAmount*numWindows, data)
}

func (l *Conv) Backward(gradOutput *mat.Dense, lr float64) *mat.Dense {
	batchSize, _ := gradOutput.Dims()
	outR := l.InR - l.KernelSize + 1
	outC := l.InC - l.KernelSize + 1
	numWindows := outR * outC
	batchScale := 1.0 / float64(batchSize)

	for k := 0; k < l.KernelsAmount; k++ {
		var db float64
		for b := 0; b < batchSize; b++ {
			row := gradOutput.RawRowView(b)
			for w := 0; w < numWindows; w++ {
				db += row[k*numWindows+w]
			}
		}
		l.Biases.Set(0, k, l.Biases.At(0, k)-lr*db*batchScale)
	}

	gradMatrix := mat.NewDense(l.KernelsAmount, batchSize*numWindows, nil)
	for b := 0; b < batchSize; b++ {
		row := gradOutput.RawRowView(b)
		for k := 0; k < l.KernelsAmount; k++ {
			for w := 0; w < numWindows; w++ {
				gradMatrix.Set(k, b*numWindows+w, row[k*numWindows+w])
			}
		}
	}

	var dXCol mat.Dense
	dXCol.Mul(l.Kernels.T(), gradMatrix)
	gradInput := im2col.FromWindowsMultiChannel(&dXCol, batchSize, l.InChannels, l.InR, l.InC, l.KernelSize)

	var dW mat.Dense
	dW.Mul(gradMatrix, l.lastIm2Col.T())
	dW.Scale(lr*batchScale, &dW)
	l.Kernels.Sub(l.Kernels, &dW)

	return gradInput
}
