package main

import (
	"fmt"
	"log"

	"github.com/olihough86/transformers-go/pkg/models/gpt2"
)

func main() {
	configPath := "../../gpt2/config.json"
	weightsPath := "../../gpt2/pytorch_model.bin"

	config, err := gpt2.LoadConfig(configPath)
	if err != nil {
		log.Fatal("Error loading config:", err)
	}

	model := gpt2.NewGPT2Model(config)

	err = model.LoadWeights(weightsPath)
	if err != nil {
		log.Fatal("Error loading weights:", err)
	}

	fmt.Println("GPT-2 model loaded successfully")
}
