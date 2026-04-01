#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M 
#SBATCH --time=2:59:00
#SBATCH --account=def-coheneld


# Don't change the lines below
#=====================================================================

module purge

module load python/3.11 scipy-stack
module load StdEnv/2023
module load tbb
module load meta-farm

source ~/envs/interns_env/bin/activate

task.run