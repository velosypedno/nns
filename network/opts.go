package network

import "go.uber.org/zap"

type Config struct {
	Logger      *zap.Logger
	LogInterval int
	BatchSize   int
	Epochs      int
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

func (n *Network) SetLogger(l *zap.Logger) {
	n.logger = l
}
