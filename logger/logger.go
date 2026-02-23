package logger

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func NewPrettyLogger() *zap.Logger {
	config := zap.NewDevelopmentConfig()

	config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	config.EncoderConfig.TimeKey = "time"
	config.EncoderConfig.EncodeTime = zapcore.TimeEncoderOfLayout("15:04:05")
	config.EncoderConfig.MessageKey = "msg"

	config.EncoderConfig.CallerKey = ""

	logger, _ := config.Build()
	return logger
}
