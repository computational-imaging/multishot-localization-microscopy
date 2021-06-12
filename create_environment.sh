#!/usr/bin/env bash
if [ "$(uname)" == "Darwin" ]; then
    conda create --name microscopy pytorch-lightning=1.2.4 python=3.8.8 pytorch=1.7 pandas trackpy scikit-image numpy scipy torchvision pytest matplotlib flake8 -c conda-forge -c pytorch
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    conda create --name microscopy pytorch-lightning=1.2.4 python=3.8.8 pytorch=1.7 cudatoolkit=11.0 pandas trackpy scikit-image numpy scipy torchvision pytest matplotlib -c pytorch
fi