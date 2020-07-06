#!/bin/sh
#SBATCH --partition=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A m2334
#SBATCH -C haswell

#SBATCH -J MFI_ethylene_ZPE_ediff
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chr218@lehigh.edu

module load vasp


date > a
srun -n 48 vasp_gam
date > b

