# NLU Course Default Final Project: GPT and Downstream Tasks

This is the default final project for the Natural Language Understanding class. 

This project comprises two parts. In the first part, you will implement some important components of the GPT-2 model to
better understand its architecture.
In the second part, you will use the token embeddings produced by your GPT-2 model on two downstream tasks: paraphrase
detection and sonnet generation. You will implement extensions to improve your model's performance on these tasks.

In broad strokes, Part 1 of this project targets:

* modules/attention.py: Missing code blocks.
* modules/gpt2_layer.py: Missing code blocks.
* models/gpt2.py: Missing code blocks.
* optimizer.py: Missing code blocks.

To test Part 1, you will run:

* `test/optimizer_test.py`: To test your implementation of `optimizer.py`.
* `test/sanity_check.py`: To test your implementation of GPT models.

In Part 2 of this project, you will train a mini-GPT to write like Shakespeare.

In Part 3 of this project, you will use GPT2 (via cloze-style classification) detect if one sentence is a paraphrase of 
another as well as generate sonnets via autoregressive language modeling.  

To test Part 3, you will run:

* `downstream-tasks/classifier.py`: To perform sentiment analysis.
* `downstream-tasks/paraphrase_detection.py`: To perform paraphrase detection. 
* `downstream-tasks/sonnet_generation.py`: To perform sonnet generation.

Important: Adjust training hyperparameters, particularly batch size, according to your GPU's specifications to optimize performance and prevent out-of-memory errors.

## Pre-testing instructions

While there are missing code blocks that you need to implement in both of these files, the main focus of this second 
part are the extensions: how you modify your GPT2 model to improve its ability to determine if one sentence is a 
paraphrase of another as well as its ability to generate sonnets. 

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies. Becareful about the working directory when you are running your scripts.

## Acknowledgement

This project is adapted from the Stanford CS 224N final project: https://github.com/cfifty/public_cs224n_gpt . 
