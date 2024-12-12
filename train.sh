#!/bin/bash
#SBATCH -c 12                                        # Number of cores (-c)
#SBATCH --gres=gpu:1                                # GPU
#SBATCH -t 0-03:00                                  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test                                 # Partition to submit to
#SBATCH --mem=64000                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o onscore_%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e onscore_%j.err
# SBATCH --constraint=a100                       # File to which STDERR will be written, %j inserts jobid
module load python/3.10.9-fasrc01 cuda/12.0.1-fasrc01 cudnn
# mamba create -n aadit_tori
mamba activate whatover
# mamba install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# pip3 install numpy, ptflops, torchjpeg, matplotlib
# pip3 install tenserflow
# mamba install tenserflow
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python rlagent.py
