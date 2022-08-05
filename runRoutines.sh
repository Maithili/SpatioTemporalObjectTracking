#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/0720_stretched/set7"

for dataset in $(find data/0720_stretched/ -mindepth 1 -maxdepth 1)
do
    echo $dataset
    for i in 1 2 3
    do
        python3 ./run.py --path=$dataset --name=ours --logs_dir=$logs_dir --write_ckpt
        for config in oneHotClassOnly  classOnly
        do
            python3 ./run.py --cfg=$config --path=$dataset --name=$config --logs_dir=$logs_dir --write_ckpt
        done
    done
    python3 ./helpers/viz.py --paths=$logs_dir
done
