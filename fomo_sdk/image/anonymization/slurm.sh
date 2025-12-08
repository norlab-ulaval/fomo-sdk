#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00
#SBATCH --job-name=ego_blur
#SBATCH --output=%x-%j.out

cd $HOME/fomo/fomo-sdk/fomo_sdk/image/anonymization || exit 1

apptainer exec --nv \
  -B ./:/mnt/egoblur \
  -B $INPUT_PATH:/input \
  -B $OUTPUT_PATH:/output \
  EgoBlur.sif \
  bash -c \
    "python3 blur_all.py \
      --input /input \
      --output /output \
      --face_model_path checkpoints/ego_blur_face.jit \
      --lp_model_path checkpoints/ego_blur_lp.jit"
