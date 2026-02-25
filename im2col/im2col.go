package im2col

import "gonum.org/v1/gonum/mat"

func ToWindows(images *mat.Dense, r, c int, kernelSize int) *mat.Dense {
	batchSize, _ := images.Dims()

	outR := r - kernelSize + 1
	outC := c - kernelSize + 1
	numWindowsPerImage := outR * outC

	rows := kernelSize * kernelSize
	cols := numWindowsPerImage * batchSize

	data := make([]float64, rows*cols)

	for b := 0; b < batchSize; b++ {
		img := images.RawRowView(b)

		for i := 0; i < outR; i++ {
			for j := 0; j < outC; j++ {

				windowIdx := b*numWindowsPerImage + i*outC + j

				for ki := 0; ki < kernelSize; ki++ {
					for kj := 0; kj < kernelSize; kj++ {
						pixel := img[(i+ki)*c+(j+kj)]

						pixelIdxInWindow := ki*kernelSize + kj
						data[pixelIdxInWindow*cols+windowIdx] = pixel
					}
				}
			}
		}
	}

	return mat.NewDense(rows, cols, data)
}

func ToWindowsMultiChannel(inputs *mat.Dense, numChannels, inR, inC, kernelSize int) *mat.Dense {
	batchSize, _ := inputs.Dims()
	outR := inR - kernelSize + 1
	outC := inC - kernelSize + 1
	numWindowsPerImage := outR * outC

	windowSize := numChannels * kernelSize * kernelSize
	totalWindows := batchSize * numWindowsPerImage

	data := make([]float64, windowSize*totalWindows)

	for b := 0; b < batchSize; b++ {
		inputRow := inputs.RawRowView(b)

		for i := 0; i < outR; i++ {
			for j := 0; j < outC; j++ {
				colIdx := b*numWindowsPerImage + i*outC + j

				for c := 0; c < numChannels; c++ {
					inChannelOffset := c * (inR * inC)
					windowChannelOffset := c * (kernelSize * kernelSize)

					for ky := 0; ky < kernelSize; ky++ {
						for kx := 0; kx < kernelSize; kx++ {
							pixelIdx := inChannelOffset + (i+ky)*inC + (j + kx)

							rowInMatrix := windowChannelOffset + ky*kernelSize + kx
							data[rowInMatrix*totalWindows+colIdx] = inputRow[pixelIdx]
						}
					}
				}
			}
		}
	}

	return mat.NewDense(windowSize, totalWindows, data)
}

func FromWindowsMultiChannel(dXCol *mat.Dense, batchSize, numChannels, inR, inC, kernelSize int) *mat.Dense {
	outR := inR - kernelSize + 1
	outC := inC - kernelSize + 1
	numWindowsPerImage := outR * outC
	inFeatures := numChannels * inR * inC

	data := make([]float64, batchSize*inFeatures)
	for b := 0; b < batchSize; b++ {
		gradInRow := data[b*inFeatures : (b+1)*inFeatures]

		for i := 0; i < outR; i++ {
			for j := 0; j < outC; j++ {
				colIdx := b*numWindowsPerImage + i*outC + j

				for c := 0; c < numChannels; c++ {
					inChannelOffset := c * (inR * inC)
					windowChannelOffset := c * (kernelSize * kernelSize)

					for ky := 0; ky < kernelSize; ky++ {
						for kx := 0; kx < kernelSize; kx++ {
							pixelIdx := inChannelOffset + (i+ky)*inC + (j + kx)

							rowInMatrix := windowChannelOffset + ky*kernelSize + kx
							val := dXCol.At(rowInMatrix, colIdx)

							gradInRow[pixelIdx] += val
						}
					}
				}
			}
		}
	}

	return mat.NewDense(batchSize, inFeatures, data)
}
