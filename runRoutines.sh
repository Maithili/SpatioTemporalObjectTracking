#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs_clusters0428"

for train_days in 50 40 30 20 10
do
    for dataset in $(find data/clusters0428/ -mindepth 1 -maxdepth 1)
    do
        # [ -d $dataset/processed ] || ~/.conda/envs/pyml/bin/python3 ./readerFileBased.py --path=$dataset
        ~/.conda/envs/pyml/bin/python3 ./run.py --cfg=default --path=$dataset --baselines --tags=$(basename $dataset)\_$DATE_TIME --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
        for i in 1 2 3 # 4 5 # 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
        do
            ~/.conda/envs/pyml/bin/python3 ./run.py --path=$dataset --name=ours --tags=$(basename $dataset)\_$DATE_TIME --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
            for config in biggerHiddenLayer bigBatch
            do
                ~/.conda/envs/pyml/bin/python3 ./run.py --cfg=$config --path=$dataset --name=ours_$config  --tags=$(basename $dataset)\_$DATE_TIME --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
            done
        done
    done
done
~/.conda/envs/pyml/bin/python3 viz.py --paths=$logs_dir/50,$logs_dir/40,$logs_dir/30,$logs_dir/20,$logs_dir/10 --combined_dir_out=$logs_dir/combined
