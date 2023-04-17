#!/bin/bash -l

# setting name of job
#SBATCH -J NEX-GDDP_nc

# setting home directory
#SBATCH -D /home/shqwu/MACA/almond_cropland_nc

# setting standard error output
#SBATCH -e /home/shqwu/NEX-GDDP/slurm_log/sterror_%j.txt

# setting standard output
#SBATCH -o /home/shqwu/NEX-GDDP/slurm_log/stdoutput_%j.txt

# setting medium priority
#SBATCH -p high2

#SBATCH --mem=128G

# setting the max time
#SBATCH -t 128:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=shqwu@ucdavis.edu




srun /home/shqwu/miniconda3/bin/python nan_no_cropland_MACA_hist.py
