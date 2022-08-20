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

for train_days in 30 35 40 45 50
do
# train_days=20
# python3 ./run.py --path=$d0 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --finetune --ckpt_dir=$logs_dir/0/persona0/ourspt --cfg=min_epochs
echo $train_days
echo $d0
python3 ./run.py --path=$d0 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --finetune --ckpt_dir=$logs_dir/0/persona0/ourspt

echo $train_days
echo $d1
python3 ./run.py --path=$d1 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --finetune --ckpt_dir=$logs_dir/0/persona1/ourspt

echo $train_days
echo $d2
python3 ./run.py --path=$d2 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --finetune --ckpt_dir=$logs_dir/0/persona2/ourspt    

echo $train_days
echo $d3
python3 ./run.py --path=$d3 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --finetune --ckpt_dir=$logs_dir/0/persona3/ourspt    

echo $train_days
echo $d4
python3 ./run.py --path=$d4 --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --finetune --ckpt_dir=$logs_dir/0/persona4/ourspt

done

