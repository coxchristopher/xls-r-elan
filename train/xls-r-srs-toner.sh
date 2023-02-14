#!/bin/bash
#SBATCH --gres=gpu:v100:1   # request one Volta V100 GPU
#SBATCH --cpus-per-task=3   # request 3 CPU cores (max for a one-GPU job)
#SBATCH --mem=64GB          # memory per node
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load StdEnv/2020 gcc/9.3.0 cuda/11.4
module load python/3.9
module load arrow

source ~/xls-r-srs/venv-xls-r-srs/bin/activate
python ~/xls-r-srs/xls-r-srs-toner.py
