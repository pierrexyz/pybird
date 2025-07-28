<<<<<<< HEAD
folder=05

#for chain in pt _pt pt_m _pt_m
for chain in boss _boss boss_m _boss_m
=======
folder=01

#for chain in boss _boss boss_m _boss_m
#for chain in boss_Geff _boss_Geff
#for chain in boss_ede _boss_ede boss_Geff _boss_Geff
#for chain in pt_m _pt_m
for chain in pt _pt
>>>>>>> 7d70553b5000c0081791cbeb06015e575e3648e4
do
	sbatch -A es_senatore slurm/sample.sh $chain $folder
	# sbatch slurm/sample.sh $chain $folder
done
