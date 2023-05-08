#!/bin/bash -l

# setting name of job
#SBATCH -J ACI_contribution

# setting home directory
#SBATCH -D 

# setting medium priority
#SBATCH -p high2

#SBATCH --mem=128G

# setting the max time
#SBATCH -t 48:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=




srun python aci_contribution.py
