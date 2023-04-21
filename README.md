# transformers-go

## How it started
Me and GPT-4 are YOLO porting transformers to Golang for a laugh, well and to try to teach me some big brain mathmatics.
Serously this will be very broken, I don't even know what a matrices is.

## Our goal
GPT-4 has proven very usful to me recently allowing me to accelerate a number of complex projects 

This is our biggest yet to have an AI write (using golang, no Pyton allowed) an implemention of Transformers and load GPT2 (initially) with a simple interface to interact with the model

## Where are we now?

Right now we have a cut down Transformers written from scratch in Golang, it seems to work, seems...

The stage we are at is actually loading a Pytorch model as a .bin, this has proven difficult due to lack of use of Go in ML / NL projects. However it's not impossible! We should be able to use [gotch](https://github.com/sugarme/gotch)

## Listen

This is not a try hard effort to port Tranformers, I have no aim to maintain a library, but this will be a fun learning experiance, I have inclued in 'the_plan' a prompt which can be use to initialize GPT-4 with the important details.

### Project tree

├── cmd
│   ├── example
│   │   └── main.go
│   └── main.go
├── go.mod
├── go.sum
├── pkg
│   └── models
│       └── gpt2
│           ├── attention.go
│           ├── config.go
│           ├── embedding_layer.go
│           ├── gpt2.go
│           ├── layer_norm.go
│           ├── loader.go
│           ├── position_wise_feed_forward.go
│           └── transformer_layer.go
├── prepostprocessors
├── README.md
├── test
│   ├── models
│   │   └── test_gpt2.go
│   ├── prepostprocessors
│   └── tokenizers
├── the_prompt
└── tokenizers

11 directories, 15 files

## The great containering

Due to needing a very stable and clean dev enronment, in the net day or so a Dockerfile will be created to keep everything nice and clean.

## How to play at home

I want you to join in, fork it, raise issues, submit PRs, but remeber this is a project where the focuse is having the AI write it's own code.
