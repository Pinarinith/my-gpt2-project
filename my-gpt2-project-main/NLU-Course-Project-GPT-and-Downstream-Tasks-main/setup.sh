#!/usr/bin/env bash

conda env create -f env.yml
conda activate nlu

export PYTHONPATH=$PYTHONPATH:$(pwd)