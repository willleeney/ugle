#!/bin/bash
 
#SBATCH --job-name=test_bp_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=150:00:0
#SBATCH --mem=25G
#SBATCH --account=EMAT022967
#SBATCH --mail-user=wl169553@bristol.ac.uk

cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"