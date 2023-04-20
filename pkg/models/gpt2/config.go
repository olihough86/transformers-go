package gpt2

import (
	"encoding/json"
	"io/ioutil"
	"os"
)

type GPT2Config struct {
    VocabSize          int    `json:"vocab_size"`
    NPositions         int    `json:"n_positions"`
    NContext           int    `json:"n_ctx"`
    NLayers            int    `json:"n_layer"`
    NHead              int    `json:"n_head"`
    HiddenSize         int    `json:"hidden_size"`
    IntermediateSize   int    `json:"intermediate_size"`
    ActivationFunction string `json:"activation_function"`
}

func LoadConfig(configFile string) (*GPT2Config, error) {
	file, err := os.Open(configFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var config Config
	err = json.Unmarshal(bytes, &config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}