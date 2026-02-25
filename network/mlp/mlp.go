package mlp

import (
	"encoding/gob"
	"fmt"
	"strings"

	"github.com/velosypedno/nns/layer"
	"github.com/velosypedno/nns/loss"

	"go.uber.org/zap"
	"gonum.org/v1/gonum/mat"
)

func init() {
	gob.Register(&layer.Dense{})
	gob.Register(&layer.Tanh{})

	gob.Register(&loss.MSE{})
	gob.Register(&loss.SoftMaxCrossEntropy{})
}

type Loss interface {
	Calculate(output, target *mat.Dense) float64
	Derivative(output, target *mat.Dense) *mat.Dense
	Transform(output *mat.Dense) *mat.Dense
}

type Layer interface {
	Forward(inputs *mat.Dense) *mat.Dense
	Backward(upstreamGradient *mat.Dense, lr float64) *mat.Dense
	fmt.Stringer
}

type MLP struct {
	Layers       []Layer
	LearningRate float64
	Loss         Loss

	logger      *zap.Logger
	logInterval int
	batchSize   int
	epochs      int
}

func New(layers []Layer, lr float64, lossFunc Loss, opts ...Option) *MLP {
	conf := &Config{
		Logger:      zap.NewNop(),
		LogInterval: 10000,
		BatchSize:   1,
		Epochs:      1000,
	}

	for _, opt := range opts {
		opt(conf)
	}
	return &MLP{
		Layers:       layers,
		LearningRate: lr,
		Loss:         lossFunc,

		logger:      conf.Logger,
		logInterval: conf.LogInterval,
		batchSize:   conf.BatchSize,
		epochs:      conf.Epochs,
	}
}

func (n *MLP) String() string {
	var sb strings.Builder
	sb.WriteString("==========================================\n")
	sb.WriteString(fmt.Sprintf("Neural Network (Learning Rate: %.4f)\n", n.LearningRate))
	sb.WriteString(fmt.Sprintf("Total Layers: %d\n", len(n.Layers)))
	sb.WriteString("==========================================\n")

	for i, layer := range n.Layers {
		sb.WriteString(fmt.Sprintf("Layer #%d ", i+1))
		sb.WriteString(layer.String())
		sb.WriteString("\n------------------------------------------\n")
	}

	return sb.String()
}

func (n *MLP) forward(inputs *mat.Dense) *mat.Dense {
	var currInputs = inputs
	for _, l := range n.Layers {
		currInputs = l.Forward(currInputs)
	}
	return currInputs
}

func (n *MLP) Predict(inputs *mat.Dense) *mat.Dense {
	logits := n.forward(inputs)
	return n.Loss.Transform(logits)
}

func (n *MLP) backward(targets, outs *mat.Dense) {
	currentGradient := n.Loss.Derivative(outs, targets)

	currentBatchSize, _ := targets.Dims()
	effectiveLR := n.LearningRate / float64(currentBatchSize)

	for i := len(n.Layers) - 1; i >= 0; i-- {
		gradPtr := n.Layers[i].Backward(currentGradient, effectiveLR)
		currentGradient.CloneFrom(gradPtr)
	}
}

func (n *MLP) Fit(X, Y *mat.Dense) {
	nSamples, nInputs := X.Dims()
	_, nOutputs := Y.Dims()

	n.logger.Info("Starting training",
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
			n.logger.Info("Training progress",
				zap.Int("epoch", e),
				zap.Float64("avg_batch_loss", epochLoss/float64(numBatches)),
			)
		}
	}
	n.logger.Info("Training complete")
}
