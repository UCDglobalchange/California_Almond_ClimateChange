#!/bin/bash -l

home_path=''

# setting name of job
#SBATCH -J MACA_projection

# setting home directory
#SBATCH -D $home_path


# setting medium priority
#SBATCH -p high2

#SBATCH --mem=124G

# setting the max time
#SBATCH -t 20:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


# send mail here
#SBATCH --mail-user=



srun python -u MACA_projection.py
