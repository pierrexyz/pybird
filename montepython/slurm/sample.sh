#!/bin/sh
<<<<<<< HEAD
#SBATCH --time=24:00:00 
#SBATCH --partition=cpu
#SBATCH --output=log/sample.o%j  # Standard output and error log

#SBATCH --mem-per-cpu=8G
=======
#SBATCH --time=12:00:00 
#SBATCH --partition=cpu
#SBATCH --output=log/sample.o%j  # Standard output and error log

#SBATCH --mem-per-cpu=2G
>>>>>>> 7d70553b5000c0081791cbeb06015e575e3648e4
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4  # This is needed for openmp

#echo "#SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#echo "#OMP_NUM_THREADS   : $OMP_NUM_THREADS"

# cd /cluster/work/senatore/pierre_bisp_montepython/
cd /cluster/work/refregier/alexree/local_packages/montepython_public
mpirun -n 16 python -u montepython/MontePython.py run -o /cluster/work/refregier/alexree/local_packages/pybird_emu/montepython/chains/$1 --superupdate 20 -f 2. -N 200000 
#-j sequential
