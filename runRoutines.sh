#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/"$DATE_TIME"_many_epochs"
tag="ManyManyEpochs"

for train_days in 50
do    
    for dataset in data/personaWithoutClothes/persona0 data/personaWithoutClothes/persona1 data/personaWithoutClothes/persona3 data/personaWithoutClothes/persona4 
    do
        dataset_name=$(basename $dataset)
        python3 ./run_activity.py --path=$dataset --name=ours --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --tags=$tag
        python3 ./run_activity.py --path=$dataset --name=original --original_model --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --tags=$tag
        for config in activity25 activity50 activity75 activity100 no_overshoot latent_simil
        do
            echo $dataset": "$config
            python3 ./run_activity.py --cfg=$config --path=$dataset --name=ours_$config --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --tags=$tag
        done
    done
done
