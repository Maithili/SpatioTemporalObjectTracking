#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/0720_stretched/set5_halfFunkyLoss"

for dataset in $(find data/0720_stretched/ -mindepth 1 -maxdepth 1)
do
    echo $dataset
    dataset_name="$(basename -- $dataset)"
    # python3 ./run.py --path=$dataset --name=ours --logs_dir=$logs_dir --read_ckpt --ckpt_dir=$logs_dir/$dataset_name/ours_30epochs
    python3 ./run.py --path=$dataset --name=ours --logs_dir=$logs_dir --read_ckpt --ckpt_dir=$logs_dir/$dataset_name/ours_20epochs
    for config in classOnly oneHotClassOnly
    do
        python3 ./run.py --cfg=$config --path=$dataset --name=$config --logs_dir=$logs_dir --read_ckpt  --ckpt_dir=$logs_dir/$dataset_name/$config\_20epochs
    done
done
