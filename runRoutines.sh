#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/probing_context"

# for train_days in 50
# do
    for dataset in $(find data/routineWithActivities/ -mindepth 1 -maxdepth 1)
    do
        echo $dataset
        # python3 ./run.py --cfg=default --path=$dataset --baselines --logs_dir=$logs_dir/$train_days --write_ckpt
        for i in 1 2 3
        do
            python3 ./run.py --path=$dataset --name=timeSine --logs_dir=$logs_dir --write_ckpt
            for config in timeLinear randomContext diffContext
            do
                python3 ./run.py --cfg=$config --path=$dataset --name=$config --logs_dir=$logs_dir --write_ckpt
            done
        done
        python3 helper/viz.py --paths=$logs_dir
    done
# done
