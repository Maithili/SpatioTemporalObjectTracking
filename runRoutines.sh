#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

for dataset in $(find data/finalDatasets0301/ -mindepth 1 -maxdepth 1)
do
    # ./readerFileBased.py --path=$dataset
    ./run.py --cfg=default --path=$dataset --baselines --tags=$(basename $dataset)\_$DATE_TIME
    for i in 1 2 3 #4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    do
        mkdir -p checkpoints/$dataset/ours
        ./run.py --path=$dataset --name=ours --tags=$(basename $dataset)\_$DATE_TIME --ckpt_dir=checkpoints/$dataset/default --write_ckpt
        for config in timeLinear allEdges #timeFewFreq timeManyFreq timeClock bigBatch bigHiddenLayer
        do
            mkdir -p checkpoints/$dataset/$config
            ./run.py --cfg=$config --path=$dataset --name=ours_$config  --tags=$(basename $dataset)\_$DATE_TIME --ckpt_dir=checkpoints/$dataset/$config --write_ckpt
        done
    done
done

python viz.py