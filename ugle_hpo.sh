#!/bin/bash

lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/default_q1.yaml

for a in sublime bgrl vgaer daegc dmon grace dgi; do
    for d in texas cora cornell wisc; do
        lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=${d}_${a}
    done
done


lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=dblp_grace
lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=dblp_bgrl
lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=dblp_dgi
lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=dblp_sublime

lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=citeseer_vgaer
lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=citeseer_daegc
lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=citeseer_dmon
lbatch -c 1 -g 1 -m 22 -t 400 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=citeseer_dgi

