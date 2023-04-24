#!/bin/bash
#SBATCH --job-name=dino-finetune
#SBATCH --nodes=2
#SBATCH --gpus-per-node=a100l.3:8
#SBATCH --mem=40GB
#SBATCH --time=50:00:00
#SBATCH --output=/home/mila/d/diganta.misra/scratch/mae_data/weights/log/out.txt
#SBATCH --error=/home/mila/d/diganta.misra/scratch/mae_data/weights/log/err.txt

module load anaconda/3

conda activate /home/mila/d/diganta.misra/.conda/envs/prompt

wandb login bd67cef57b7227730fe3edf96e11d954558a9d0d

ulimit -Sn $(ulimit -Hn)

python run_with_submitit.py --nodes 2 --ngpus 8 --use_volta32 --arch vit_base  --data_path /home/mila/d/diganta.misra/scratch/imagenet/train --output_dir /home/mila/d/diganta.misra/scratch/mae_data/weights/log --pretrained_weights /home/mila/d/diganta.misra/scratch/mae_data/weights/dino_vitbase16_pretrain.pth