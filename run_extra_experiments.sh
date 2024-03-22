#!/bin/bash

for a in sublime vgaer; do

    for d in bat citeseer cora dblp eat texas wisc cornell uat amac amap; do
        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/complex_networks/default.yaml -da=${d}_${a}_default
        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/complex_networks/hpo.yaml -da=${d}_${a}
    end

    for d in cornell wisc texas dblp citeseer cora; do
        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/33_train_sup.yaml -da=${d}_${a}_default
        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/33_train.yaml -da=${d}_${a}_default
        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/66_train_sup.yaml -da=${d}_${a}_default
        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/66_train.yaml -da=${d}_${a}_default

        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/default_sup.yaml -da=${d}_${a}_default
        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/default.yaml -da=${d}_${a}_default

        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/hpo_sup.yaml -da=${d}_${a}
        lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/hpo.yaml -da=${d}_${a}
    end

    lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/synth.yaml -da=${d}_${a}_default
    lbatch -c 1 -g 1 -m 22 -t 300 -a EMAT022967 -q gpu --conda-env UFR --cmd python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/synth_sup.yaml -da=${d}_${a}_default
end