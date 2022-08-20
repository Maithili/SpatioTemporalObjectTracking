#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/CoRL_eval_"$DATE_TIME

for train_days in 50 40 30 20 10
do
    for dataset in $(find data/personaWithoutClothes/ -mindepth 1 -maxdepth 1)
    do
        echo $dataset
        python3 ./run.py --path=$dataset --name=ours --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
    done
done

train_days=50
for dataset in $(find data/personaWithoutClothes/ -mindepth 1 -maxdepth 1)
do
    for config in allEdges timeLinear move2 move3
    do
        python3 ./run.py --cfg=$config --path=$dataset --name=ours_$config --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
    done
done

for train_days in 50 40 30 20 10
for dataset in $(find data/personaWithoutClothes/ -mindepth 1 -maxdepth 1)
    do
        echo $dataset
        python3 ./run.py --cfg=default --path=$dataset --baselines --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
    done
done

for train_days in 5 15 25 35 45
do
    for dataset in $(find data/personaWithoutClothes/ -mindepth 1 -maxdepth 1)
    do
        echo $dataset
        python3 ./run.py --cfg=default --path=$dataset --baselines --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
        python3 ./run.py --path=$dataset --name=ours --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
    done
done
