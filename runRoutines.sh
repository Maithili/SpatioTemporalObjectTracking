#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/"$DATE_TIME"_object_activity"

for train_days in 50 40 30 20 10 5 15 25 35 45
do    
    for dataset in  $(find data/personaWithoutClothes/ -mindepth 1 -maxdepth 1)
    do
        dataset_name=$(basename $dataset)
        python3 ./run_activity.py --path=$dataset --name=ours --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
        for config in activity25 activity50 activity75 activity100
        do
            echo $dataset": "$config
            python3 ./run_activity.py --cfg=$config --path=$dataset --name=ours_$config --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
        done
    done
done
