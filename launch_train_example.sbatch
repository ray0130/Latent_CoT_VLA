#!/bin/bash
#
#SBATCH --job-name=train_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --account=<SLURM ACCOUNT>
#SBATCH --partition=<GPU PARTITION>
#SBATCH --requeue

export SINGULARITY_BINDPATH=/home,/scratch,/share/apps


singularity exec --nv \
            --overlay your/overlay/image:rw \
            your/cuda/sif/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
            /bin/bash -c "
            unset -f which;
            source /ext3/env.sh;
            conda activate cv_env;
            export WANDB_API_KEY=\"YOUR_WANDB_API_KEY\";
            export WANDB_MODE=online;
            which python;
            nvidia-smi;
            cd /your/project/folder/path/Latent_CoT_VLA;
            ./scripts/train/train_latent_cotvla.sh
            "