package gpt2

import (
	"encoding/json"
	"io/ioutil"
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

func LoadConfig(path string) (*GPT2Config, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config GPT2Config
	err = json.Unmarshal(data, &config)
	if err != nil {
		return nil, err
	} 

	return &config, nil
}