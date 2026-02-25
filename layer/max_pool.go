package layer

import "gonum.org/v1/gonum/mat"

type MaxPool struct {
	Size   int
	Stride int

	InChannels int
	InR, InC   int

	maxIndices []int
}

func NewMaxPool(size, stride, inChannels, inR, inC int) *MaxPool {
	return &MaxPool{
		Size:       size,
		Stride:     stride,
		InChannels: inChannels,
		InR:        inR,
		InC:        inC,
	}
}

func (l *MaxPool) Forward(inputs *mat.Dense) *mat.Dense {
	batchSize, _ := inputs.Dims()

	outR := (l.InR-l.Size)/l.Stride + 1
	outC := (l.InC-l.Size)/l.Stride + 1

	outFeatures := l.InChannels * outR * outC
	data := make([]float64, batchSize*outFeatures)

	l.maxIndices = make([]int, batchSize*outFeatures)

	for b := 0; b < batchSize; b++ {
		inputRow := inputs.RawRowView(b)
		outputRow := data[b*outFeatures : (b+1)*outFeatures]

		for c := 0; c < l.InChannels; c++ {
			inChannelOffset := c * (l.InR * l.InC)
			outChannelOffset := c * (outR * outC)

			for i := 0; i < outR; i++ {
				for j := 0; j < outC; j++ {
					inYStart := i * l.Stride
					inXStart := j * l.Stride

					maxVal := -1e99
					maxIdx := -1

					for ky := 0; ky < l.Size; ky++ {
						for kx := 0; kx < l.Size; kx++ {
							pixelIdx := inChannelOffset + (inYStart+ky)*l.InC + (inXStart + kx)
							val := inputRow[pixelIdx]
							if val > maxVal {
								maxVal = val
								maxIdx = pixelIdx
							}
						}
					}

					outIdx := outChannelOffset + i*outC + j
					outputRow[outIdx] = maxVal
					l.maxIndices[b*outFeatures+outIdx] = maxIdx
				}
			}
		}
	}

	return mat.NewDense(batchSize, outFeatures, data)
}

func (l *MaxPool) Backward(gradOutput *mat.Dense, lr float64) *mat.Dense {
	batchSize, outFeatures := gradOutput.Dims()
	inFeatures := l.InChannels * l.InR * l.InC

	gradInputData := make([]float64, batchSize*inFeatures)

	for b := 0; b < batchSize; b++ {
		gradOutRow := gradOutput.RawRowView(b)
		gradInRow := gradInputData[b*inFeatures : (b+1)*inFeatures]

		for outIdx := 0; outIdx < outFeatures; outIdx++ {
			maskIdx := b*outFeatures + outIdx
			originalPixelIdx := l.maxIndices[maskIdx]

			gradInRow[originalPixelIdx] += gradOutRow[outIdx]
		}
	}

	return mat.NewDense(batchSize, inFeatures, gradInputData)
}
