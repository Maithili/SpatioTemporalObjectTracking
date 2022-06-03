#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs"

for train_days in 50
do
    for dataset in $(find data/persona/ -mindepth 1 -maxdepth 1)
    do
        echo $dataset
        python3 ./run.py --cfg=default --path=$dataset --baselines --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
        for i in 1 2 3
        do
            python3 ./run.py --path=$dataset --name=ours --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
            for config in allEdges timeLinear
            do
                python3 ./run.py --cfg=$config --path=$dataset --name=ours_$config --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
            done
        done
    done
    python3 viz.py --paths=$logs_dir/$train_days
done
