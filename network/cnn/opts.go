package cnn

import (
	"go.uber.org/zap"
)

type Config struct {
	Logger       *zap.Logger
	LogInterval  int
	BatchSize    int
	Epochs       int
	LearningRate float64
	Loss         Loss
}

type Option func(*Config)

func WithLogger(logger *zap.Logger) Option {
	return func(c *Config) {
		c.Logger = logger
	}
}

func WithLogInterval(interval int) Option {
	return func(c *Config) {
		c.LogInterval = interval
	}
}

func WithBatchSize(size int) Option {
	return func(c *Config) {
		c.BatchSize = size
	}
}

func WithEpochs(epochs int) Option {
	return func(c *Config) {
		c.Epochs = epochs
	}
}

func WithLearningRate(lr float64) Option {
	return func(c *Config) {
		c.LearningRate = lr
	}
}

func WithLoss(l Loss) Option {
	return func(c *Config) {
		c.Loss = l
	}
}
