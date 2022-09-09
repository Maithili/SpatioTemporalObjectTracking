#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/"$DATE_TIME"_CoRL_final"

train_days=50
for dataset in $(find data/personaWithoutClothesAllObj/ -mindepth 1 -maxdepth 1)
do
    echo $dataset": default"
    python3 ./run.py --cfg=confidence --path=$dataset --name=ours --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
    echo $dataset": baselines"
    python3 ./run.py --cfg=default --path=$dataset --baselines --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
    for config in allEdges timeLinear
    do
        echo $dataset": "$config
        python3 ./run.py --cfg=$config --path=$dataset --name=ours_$config --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
    done
done

for train_days in 40 30 20 10 5 15 25 35 45
do
    for dataset in $(find data/personaWithoutClothesAllObj/ -mindepth 1 -maxdepth 1)
    do
        echo $dataset": default"
        python3 ./run.py --path=$dataset --name=ours --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
        echo $dataset": baselines"
        python3 ./run.py --cfg=default --path=$dataset --baselines --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
    done
done
