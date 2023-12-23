#!/bin/bash

lbatch -c 1 -g 1 -m 22 -t 200 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/default_q1.yaml

for a in sublime bgrl vgaer daegc dmon grace dgi; do
    for d in citeseer texas cora dblp cornell wisc; do
        lbatch -c 1 -g 1 -m 22 -t 200 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=${d}_${a}
    done
done

lbatch -c 1 -g 1 -m 30 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/dmon_large_computers.yaml
lbatch -c 1 -g 1 -m 30 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/dmon_large_photos.yaml

