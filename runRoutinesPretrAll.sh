#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

DATE_TIME=`date "+%m%d_%H%M"`

logs_dir="logs/CoRL_eval_allobjdata"

d0="data/persona_allobj/persona0"
d1="data/persona_allobj/persona1"
d2="data/persona_allobj/persona2"
d3="data/persona_allobj/persona3"
d4="data/persona_allobj/persona4"

train_days=0

python3 ./run_pt.py --path=$d2 --pretrain_dirs=$d0,$d1,$d3,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
python3 ./run_pt.py --path=$d3 --pretrain_dirs=$d0,$d1,$d2,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
python3 ./run_pt.py --path=$d4 --pretrain_dirs=$d0,$d1,$d2,$d3 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt

python3 ./run_pt.py --path=$d0 --pretrain_dirs=$d1,$d2,$d3,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
python3 ./run_pt.py --path=$d1 --pretrain_dirs=$d0,$d2,$d3,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt


# echo $train_days
# echo $d2
# python3 ./run_pt.py --path=$d2 --pretrain_dirs=$d0,$d1,$d3,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
# echo $train_days
# echo $d3
# python3 ./run_pt.py --path=$d3 --pretrain_dirs=$d0,$d1,$d2,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
# echo $train_days
# echo $d4
# python3 ./run_pt.py --path=$d4 --pretrain_dirs=$d0,$d1,$d2,$d3 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt


# for train_days in 20 25
# do
#     echo $train_days
#     echo $d0
#     python3 ./run_pt.py --path=$d0 --pretrain_dirs=$d1,$d2,$d3,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
#     echo $train_days
#     echo $d1
#     python3 ./run_pt.py --path=$d1 --pretrain_dirs=$d0,$d2,$d3,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
#     echo $train_days
#     echo $d2
#     python3 ./run_pt.py --path=$d2 --pretrain_dirs=$d0,$d1,$d3,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
#     echo $train_days
#     echo $d3
#     python3 ./run_pt.py --path=$d3 --pretrain_dirs=$d0,$d1,$d2,$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt
#     echo $train_days
#     echo $d4
#     python3 ./run_pt.py --path=$d4 --pretrain_dirs=$d0,$d1,$d2,$d3 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt


#     python3 helpers/viz.py --paths=$logs_dir/$train_days
# done
