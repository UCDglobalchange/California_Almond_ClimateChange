#!/bin/bash -l

# setting name of job
#SBATCH -J Lasso_model

# setting home directory
#SBATCH -D

# setting medium priority
#SBATCH -p high2

# setting standard error output
#SBATCH -e 

# setting standard output
#SBATCH -o 

#SBATCH --mem=1G

# setting the max time
#SBATCH -t 20:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

#SBATCH --array=1-1000

# send mail here
#SBATCH --mail-user=



srun python -u Almond_lasso.py $SLURM_ARRAY_TASK_ID
