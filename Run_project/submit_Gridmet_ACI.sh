#!/bin/bash -l

# setting name of job
#SBATCH -J Gridmet_ACI

# setting home directory
#SBATCH -D 

# setting medium priority
#SBATCH -p high2

# setting standard error output
#SBATCH -e 

# setting standard output
#SBATCH -o 

#SBATCH --mem=64G

# setting the max time
#SBATCH -t 10:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=

srun python  Gridmet_ACI.py
