package cnn

import (
	"encoding/gob"
	"fmt"

	"github.com/velosypedno/nns/layer"
	"github.com/velosypedno/nns/loss"
	"go.uber.org/zap"
	"gonum.org/v1/gonum/mat"
)

func init() {
	gob.Register(&layer.Dense{})
	gob.Register(&layer.Tanh{})
	gob.Register(&layer.Conv{})
	gob.Register(&layer.MaxPool{})
	gob.Register(&layer.ReLU{})

	gob.Register(&loss.MSE{})
	gob.Register(&loss.SoftMaxCrossEntropy{})
}

type CNNLayer interface {
	Forward(inputs *mat.Dense) *mat.Dense
	Backward(upstreamGradient *mat.Dense, lr float64) *mat.Dense
}

type MLPLayer interface {
	Forward(inputs *mat.Dense) *mat.Dense
	Backward(upstreamGradient *mat.Dense, lr float64) *mat.Dense
	fmt.Stringer
}

type Loss interface {
	Calculate(output, target *mat.Dense) float64
	Derivative(output, target *mat.Dense) *mat.Dense
	Transform(output *mat.Dense) *mat.Dense
}

type CNN struct {
	ConvLayers       []CNNLayer
	ClassifierLayers []MLPLayer

	logger       *zap.Logger
	logInterval  int
	batchSize    int
	LearningRate float64
	Loss         Loss
	epochs       int
}

func New(convLayers []CNNLayer, classifierLayers []MLPLayer, opts ...Option) *CNN {
	conf := &Config{
		Logger:       zap.NewNop(),
		LogInterval:  100,
		BatchSize:    1,
		Epochs:       10,
		LearningRate: 0.01,
		Loss:         nil,
	}

	for _, opt := range opts {
		opt(conf)
	}

	return &CNN{
		ConvLayers:       convLayers,
		ClassifierLayers: classifierLayers,
		logger:           conf.Logger,
		logInterval:      conf.LogInterval,
		batchSize:        conf.BatchSize,
		epochs:           conf.Epochs,
		LearningRate:     conf.LearningRate,
		Loss:             conf.Loss,
	}
}

func (n *CNN) forward(inputs *mat.Dense) *mat.Dense {
	var currInputs = inputs
	for _, l := range n.ConvLayers {
		currInputs = l.Forward(currInputs)
	}

	for _, l := range n.ClassifierLayers {
		currInputs = l.Forward(currInputs)
	}

	return currInputs
}

func (n *CNN) Predict(inputs *mat.Dense) *mat.Dense {
	logits := n.forward(inputs)
	return n.Loss.Transform(logits)
}

func (n *CNN) backward(targets, outs *mat.Dense) {
	currentGradient := n.Loss.Derivative(outs, targets)

	currentBatchSize, _ := targets.Dims()
	effectiveLR := n.LearningRate / float64(currentBatchSize)

	for i := len(n.ClassifierLayers) - 1; i >= 0; i-- {
		gradPtr := n.ClassifierLayers[i].Backward(currentGradient, effectiveLR)
		currentGradient.CloneFrom(gradPtr)
	}

	for i := len(n.ConvLayers) - 1; i >= 0; i-- {
		currentGradient = n.ConvLayers[i].Backward(currentGradient, n.LearningRate)
	}
}

func (n *CNN) Fit(X, Y *mat.Dense) {
	nSamples, nInputs := X.Dims()
	_, nOutputs := Y.Dims()

	n.logger.Info("Starting CNN training",
		zap.Int("epochs", n.epochs),
		zap.Int("samples", nSamples),
		zap.Int("batch_size", n.batchSize),
		zap.Float64("lr", n.LearningRate),
	)

	for e := 0; e < n.epochs; e++ {
		epochLoss := 0.0
		numBatches := 0

		for i := 0; i < nSamples; i += n.batchSize {
			end := i + n.batchSize
			if end > nSamples {
				end = nSamples
			}

			batchX := X.Slice(i, end, 0, nInputs).(*mat.Dense)
			batchY := Y.Slice(i, end, 0, nOutputs).(*mat.Dense)

			output := n.forward(batchX)

			n.backward(batchY, output)

			epochLoss += n.Loss.Calculate(output, batchY)
			numBatches++
		}

		if n.logInterval > 0 && e%n.logInterval == 0 {
			n.logger.Info("Epoch progress",
				zap.Int("epoch", e),
				zap.Float64("avg_batch_loss", epochLoss/float64(numBatches)),
			)
		}
	}
	n.logger.Info("Training complete")
}
