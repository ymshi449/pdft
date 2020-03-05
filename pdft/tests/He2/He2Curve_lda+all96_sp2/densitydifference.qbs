#!/bin/sh -l
# FILENAME:  densitydifference.qbs

#SBATCH -A awasser 
#SBATCH -N 1
#SBATCH -t 70:00:00
#SBATCH --job-name=He2Curve_lda+all96_sp2
#SBATCH -e error.dat
#SBATCH -o output.dat

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

