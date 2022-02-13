#!/bin/bash


for dataset in persona0212/persona0212_hard_worker persona0212/persona0212_home_maker persona0212/persona0212_senior persona0212/persona0212_work_from_home
do
    ./run.py --cfg=default --path=data/$dataset --baselines
    for config in default edgeAll edgeExist noDuplicationPenalty noDropout # timeSimple learnContext 
        do
        echo "Running dataset "$dataset" with config "$config"..."
        for i in 1 2 3 # 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
        do
            ./run.py --cfg=$config --path=data/$dataset --name=$config
        done
    done
done