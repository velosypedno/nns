package cnn

import (
	"encoding/gob"
	"io"
	"os"

	"go.uber.org/zap"
)

func (n *CNN) Save(w io.Writer) error {
	encoder := gob.NewEncoder(w)
	return encoder.Encode(n)
}

func Load(r io.Reader) (*CNN, error) {
	var n CNN
	decoder := gob.NewDecoder(r)
	err := decoder.Decode(&n)
	if err != nil {
		return nil, err
	}
	n.logger = zap.NewNop()
	return &n, nil
}

func (n *CNN) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	return n.Save(file)
}

func LoadFromFile(filename string) (*CNN, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return Load(file)
}
