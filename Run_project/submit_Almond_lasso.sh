#!/bin/bash -l

home_path=''

# setting name of job
#SBATCH -J Lasso_model

# setting home directory
#SBATCH -D $home_path

# setting medium priority
#SBATCH -p high2

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
