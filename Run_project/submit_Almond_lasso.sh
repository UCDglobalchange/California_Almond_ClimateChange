#!/bin/bash -l

# setting name of job
#SBATCH -J autogluon

# setting home directory
#SBATCH -D /home/shqwu/Almond_code_git

# setting standard error output
#SBATCH -e /home/shqwu/NEX-GDDP/slurm_log/sterror_%j.txt

# setting standard output
#SBATCH -o /home/shqwu/NEX-GDDP/slurm_log/stdoutput_%j.txt

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



srun /home/shqwu/miniconda3/bin/python -u Almond_lasso.py $SLURM_ARRAY_TASK_ID
