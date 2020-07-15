#!/bin/bash
## A template to generate all slurm files

#SBATCH --job-name=egta
#SBATCH --mail-user=qmaai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=13g
#SBATCH --time=04-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard
