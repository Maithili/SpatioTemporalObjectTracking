#!/bin/bash

DATE_TIME=`date "+%m%d_%H%M"`

for dataset in $(find data/Persona0219/ -mindepth 1 -maxdepth 1)
do
    # ./readerFileBased.py --path=$dataset
    ./run.py --cfg=default --path=$dataset --baselines --tags=$(basename $dataset)\_$DATE_TIME
    for i in 1 2 3 # 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    do
        mkdir -p checkpoints/$dataset/ours
        ./run.py --path=$dataset --name=ours --ckpt_dir=checkpoints/$dataset/default --write_ckpt --tags=$(basename $dataset)\_$DATE_TIME
        for config in duplicationPenalty bigHiddenLayer
        do
            mkdir -p checkpoints/$dataset/$config
            ./run.py --cfg=$config --path=$dataset --name=ours_$config --ckpt_dir=checkpoints/$dataset/$config --write_ckpt --tags=$(basename $dataset)\_$DATE_TIME
        done
    done
done

