#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M 
#SBATCH --time=2:55:00
#SBATCH --account=def-coheneld

# Don't change the lines below
#=====================================================================

module purge
module load StdEnv/2023
module load meta-farm
module load python/3.11 scipy-stack
module load tbb
module load rust

source ~/envs/intern_env/bin/activate

task.run