#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --partition gpu
#SBATCH --job-name=ugle_hpo
#SBATCH --account=EMAT022967

conda activate ugle
python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/dmon_large_computers.yaml
python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/dmon_large_photos.yaml
