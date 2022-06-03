#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT


for dataset in $(find data/persons0509/ -mindepth 1 -maxdepth 1)
do
    echo $dataset
    [ -f $dataset ] || ~/.conda/envs/pyml/bin/python3 ./reader.py --path=$dataset
done