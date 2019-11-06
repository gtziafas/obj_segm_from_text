#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
module load PyTorch 
module load pip 
pip install torch torchvision tensorboard dataclasses tqdm fastprogress yacs spacy fire --user
python3 -m spacy download en_core_web_md --user

python3 main_dist.py "try_1" --ds_to_use='flickr30k' --bs=16 --nw=1 > console_output.txt

