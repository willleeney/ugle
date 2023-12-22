#!/bin/bash

for a in sublime bgrl vgaer daegc dmon grace dgi; do
    for d in citeseer texas cora dblp cornell wisc; do
        echo -c 1 -g 1 -m 22 -a EMAT022967 -q gpu --conda-env UFR --cmd  python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=${d}_${a}
    done
done

echo -c 1 -g 1 -m 30 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/dmon_large_computers.yaml
echo -c 1 -g 1 -m 30 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/dmon_large_photos.yaml

