#!/bin/bash
#BSUB -J sapiens_train	 		# Job name
#BSUB -n 1             			# Number of tasks
#BSUB -q long 				# Select queue
#BSUB -R "rusage[mem=8GB]"		# Memory
#BSUB -e ./logs/output_%J.err 		# Error file
#BSUB -N 				# Notify completion via email
#BSUB -gpu "num=1:j_exclusive=yes"      # 1 GPU 
# -o ./logs/output_%J.out 		# Output file (add BSUB)

ml Anaconda3
conda activate sapiens
commit=$(git log --pretty=format:'%h' -n 1)
python sapiens/train.py --logdir=resources/logdir/$commit --checkpoint=../../scratch/checkpoints/$commit/
