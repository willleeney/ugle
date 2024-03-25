#!/bin/bash


for d in bat citeseer cora dblp eat texas wisc cornell uat amac amap; do
    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/complex_networks/default.yaml -da=${d}_vgaer_default --gpu=0
    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/complex_networks/hpo.yaml -da=${d}_vgaer --gpu=0
done

for d in cornell wisc texas dblp citeseer cora; do
    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/33_train_sup.yaml -da=${d}_vgaer_default --gpu=0
    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/33_train.yaml -da=${d}_vgaer_default --gpu=0
    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/66_train_sup.yaml -da=${d}_vgaer_default --gpu=0
    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/66_train.yaml -da=${d}_vgaer_default --gpu=0

    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/default_sup.yaml -da=${d}_vgaer_default --gpu=0
    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/default.yaml -da=${d}_vgaer_default --gpu=0

    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/hpo_sup.yaml -da=${d}_vgaer --gpu=0
    python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/hpo.yaml -da=${d}_vgaer --gpu=0
done

python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/synth.yaml -da=${d}_vgaer_default --gpu=0
python3 model_evaluations.py -ec==ugle/configs/experiments/paper_configs/unsupervised_limit/synth_sup.yaml -da=${d}_vgaer_default --gpu=0

