#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p arm --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --job-name=GH-TEST
#SBATCH --time=72:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Attention

# Setup conda
source ~/.bashrc
conda activate mingi-arm

python main.py --device cuda --output_dir ./Final_Output/Test_Run