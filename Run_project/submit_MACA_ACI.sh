#!/bin/bash -l

# setting name of job
#SBATCH -J 'MRI-CGCM3'

# setting home directory
#SBATCH -D $home_path

# setting medium priority
#SBATCH -p high2

#SBATCH --mem=64G

# setting the max time
#SBATCH -t 20:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=

srun  python -u  MACA_ACI.py
