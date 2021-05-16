#!/bin/bash

#SBATCH --nodes=1  #Allocate whatever you need here
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1  #Allocate whatever you need here
#SBATCH --output=run.out
#SBATCH --job-name=test
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=aipi0122@colorado.edu
#SBATCH --mail-type=ALL

module purge
source /curc/sw/anaconda3/2019.07/bin/activate
conda activate cities
python -u pezTest.py
