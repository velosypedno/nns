package mlp

import (
	"encoding/gob"
	"io"
	"os"

	"go.uber.org/zap"
)

func (n *MLP) Save(w io.Writer) error {
	encoder := gob.NewEncoder(w)
	return encoder.Encode(n)
}

func Load(r io.Reader) (*MLP, error) {
	var n MLP
	decoder := gob.NewDecoder(r)
	err := decoder.Decode(&n)
	if err != nil {
		return nil, err
	}
	n.logger = zap.NewNop()
	return &n, nil
}

func (n *MLP) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	return n.Save(file)
}

func LoadFromFile(filename string) (*MLP, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return Load(file)
}
