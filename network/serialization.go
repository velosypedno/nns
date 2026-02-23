package network

import (
	"encoding/gob"
	"io"
	"os"

	"go.uber.org/zap"
)

func (n *Network) Save(w io.Writer) error {
	encoder := gob.NewEncoder(w)
	return encoder.Encode(n)
}

func LoadNetwork(r io.Reader) (*Network, error) {
	var n Network
	decoder := gob.NewDecoder(r)
	err := decoder.Decode(&n)
	if err != nil {
		return nil, err
	}
	n.logger = zap.NewNop()
	return &n, nil
}

func (n *Network) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	return n.Save(file)
}

func LoadFromFile(filename string) (*Network, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return LoadNetwork(file)
}
