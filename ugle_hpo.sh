#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --partition gpu
#SBATCH --job-name=ugle_hpo
#SBATCH --account=EMAT022967

conda activate ugle
for a in sublime bgrl vgaer daegc dmon grace dgi; do
    for d in citeseer texas cora dblp cornell wisc; do
        python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -ad=${d}_${a}
    done
done
