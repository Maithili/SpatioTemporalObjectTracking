#!/bin/bash

for dataset in breakfast breakfast50 pilot0117 pilot0117_allObj
do
    ./run.py --path=data/$dataset --name=$dataset
    for config in edgeAll edgeExist timeSine dropout noEdgeInputs onlyConfidentActions stochasticLoop
    do
        ./run.py --cfg=$config --path=data/$dataset --name=$config
    done
done