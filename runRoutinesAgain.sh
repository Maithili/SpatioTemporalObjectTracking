#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/CoRL_eval_0819_2204"
checkpoints_source="logs_archived/logs_persons0509"

for train_days in 5 10 15 20 25 30 35 40 45 50
do
    for dataset in $(find data/persona/ -mindepth 1 -maxdepth 1)
    do
        echo $dataset
        dataset_name=$(basename $dataset)
        python3 ./run.py --cfg=default --path=$dataset --baselines --train_days=$train_days --logs_dir=$logs_dir/$train_days
        # python3 ./run.py --path=$dataset --name=ours_50epochs --train_days=$train_days --logs_dir=$logs_dir/$train_days --read_ckpt --ckpt_dir=$checkpoints_source/$train_days/$dataset_name/ours_50epochs
        # for i in 1 2
        # do
        #     python3 ./run.py --path=$dataset --name=ours_50epochs_$i --train_days=$train_days --logs_dir=$logs_dir/$train_days --read_ckpt --ckpt_dir=$checkpoints_source/$train_days/$dataset_name/ours_50epochs_$i
            # for config in allEdges timeLinear
            # do
                # python3 ./run.py --cfg=$config --path=$dataset --name=ours_$config"_50epochs" --train_days=$train_days --logs_dir=$logs_dir/$train_days --read_ckpt --ckpt_dir=$checkpoints_source/$train_days/$dataset_name/ours_$config"_50epochs"
            # done
        done
    done
done
