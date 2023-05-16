#!/bin/bash -l

# setting name of job
#SBATCH -J mask_Cropland

# setting home directory
#SBATCH -D /home/shqwu/California_Almond_ClimateChange-main/almond_cropland_nc

# setting standard error output
#SBATCH -e /home/shqwu/NEX-GDDP/slurm_log/sterror_%j.txt

# setting standard output
#SBATCH -o /home/shqwu/NEX-GDDP/slurm_log/stdoutput_%j.txt

# setting medium priority
#SBATCH -p high2

#SBATCH --mem=64G

# setting the max time
#SBATCH -t 10:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=

srun python  MACA_mask_previous.py
