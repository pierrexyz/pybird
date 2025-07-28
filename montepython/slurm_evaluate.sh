<<<<<<< HEAD
folder=05

for chain in pt _pt pt_m _pt_m
#for chain in boss _boss boss_m _boss_m
=======
folder=01

#for chain in _boss_Geff boss_Geff
for chain in pt _pt
>>>>>>> 7d70553b5000c0081791cbeb06015e575e3648e4
do
	sbatch -A es_senatore slurm/evaluate.sh $chain $folder
#	sbatch slurm/evaluate.sh $chain $folder
done
