#!/bin/bash


for dataset in  persona0212/persona0212_senior persona0212/persona0212_work_from_home persona0212/persona0212_hard_worker persona0212/persona0212_home_maker
do
    # ./run.py --cfg=default --path=data/$dataset --baselines
    for i in 1 2 3 # 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    do
        for base in oneStepPredicted twoStepExisting
        do
            ./run.py --architecture_cfg=$base --path=data/$dataset --name=$base
            for config in timePeriodLearned noDuplicationPenalty lowDuplicationPenalty noDropout
            do
                ./run.py --architecture_cfg=$base --cfg=$config --path=data/$dataset --name=$base\_$config
            done
        done
    done
done