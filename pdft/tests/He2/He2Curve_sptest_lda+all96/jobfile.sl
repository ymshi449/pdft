#!/bin/sh -l
# FILENAME:  jobfile.sl

#SBATCH -A awasser 
#SBATCH -N 1
#SBATCH -t 140:00:00
#SBATCH --job-name=He2Curve_sptest_lda+all96
#SBATCH -e error.dat
#SBATCH -o output.dat
#SBATCH --mem=70G

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR

# Conda env
module purge
module load anaconda
module load use.own
source activate p4env
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
python He2Curve.py
echo "Job done at $(date)" >> log

