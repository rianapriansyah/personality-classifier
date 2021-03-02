#!/bin/bash
#
#SBATCH --job-name=train_w2v
#SBATCH --output=train_w2v.txt
#
#SBATCH --nodes=1
#SBATCH --time=4320:00

srun python wiki_data_prep.py
