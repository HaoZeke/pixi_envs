#!/bin/bash
#SBATCH --job-name=lmp-//SOL//
#SBATCH --output=slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem-per-gpu=90G
#SBATCH --time 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100

module purge
module load gcc
module load openmpi/5.0.3-cuda

lmp=~/repos/lab-cosmo/lammps/build/lmp
srun $lmp -in lmp.in -k on g 1
