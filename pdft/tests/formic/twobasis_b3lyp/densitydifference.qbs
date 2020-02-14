#!/bin/sh -l
# FILENAME:  densitydifference.qbs

#PBS -q awasser 
#PBS -l nodes=1:ppn=20,walltime=70:00:00
#PBS -N formic_svd1e-34567_l0_b3lyp
#PBS -e error.dat
#PBS -o output.dat

echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Conda env
module purge
module load anaconda
module load use.own
source activate p4env
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
python formic.py
echo "Job done at $(date)" >> log

