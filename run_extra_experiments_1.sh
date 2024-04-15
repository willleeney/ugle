#!/bin/bash

for d in bat citeseer cora dblp eat texas wisc cornell uat amac amap; do
    lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/complex_networks/default.yaml -da=${d}_sublime_default --gpu=0
    lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/complex_networks/hpo.yaml -da=${d}_sublime --gpu=0
done

# for d in cornell wisc texas dblp citeseer cora; do
#     lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/33_train_sup.yaml -da=${d}_sublime_default --gpu=0
#     lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/33_train.yaml -da=${d}_sublime_default --gpu=0
#     lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/66_train_sup.yaml -da=${d}_sublime_default --gpu=0
#     lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/66_train.yaml -da=${d}_sublime_default --gpu=0

#     lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/default_sup.yaml -da=${d}_sublime_default --gpu=0
#     lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/default.yaml -da=${d}_sublime_default --gpu=0

#     lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/hpo_sup.yaml -da=${d}_sublime --gpu=0
#     lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/hpo.yaml -da=${d}_sublime --gpu=0
# done

# lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/synth.yaml -da=${d}_sublime_default --gpu=0
# lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec=ugle/configs/experiments/paper_configs/unsupervised_limit/synth_sup.yaml -da=${d}_sublime_default --gpu=0

