#!/bin/bash
#SBATCH --job-name="flick_1"
#SBATCH --partition=gpu
#SBATCh --time=06:00:00
#SBATCH --gres=gpu:v100:1
module load pip PyTorch CUDA cuDNN
pip install torch torchvision tensorboard dataclasses tqdm fastprogress yacs spacy fire --user
python3 -m spacy download en_core_web_md --user

python3 code/main_dist.py "try_1" --ds_to_use='flickr30k' --bs=2 --nw=0 |& tee console_output.txt

