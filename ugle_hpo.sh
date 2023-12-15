#!/bin/bash

for a in sublime bgrl vgaer daegc dmon grace dgi; do
    for d in citeseer texas cora dblp cornell wisc; do
        lbatch -c 1 -g 1 -m 22 -a EMAT022967 -q ugle_hpo --conda-env ugle --cmd  python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -ad=${d}_${a}
    done
done

lbatch -c 1 -g 1 -m 30 -a EMAT022967 -q gpu --conda-env ugle --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/dmon_large_computers.yaml
lbatch -c 1 -g 1 -m 30 -a EMAT022967 -q gpu --conda-env ugle --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/dmon_large_photos.yaml

