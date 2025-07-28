#!/bin/sh
#SBATCH --time=01:00:00 
#SBATCH --partition=cpu
#SBATCH --output=log/evaluate.o%j  # Standard output and error log

#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1  # This is needed for openmp

cd /cluster/work/senatore/pierre_bisp_montepython/

python -u montepython/MontePython.py run -o /cluster/work/senatore/pybird_emu/montepython/chains/$2/$1 -f 0 #--bestfit chains/$1/results.minimized
