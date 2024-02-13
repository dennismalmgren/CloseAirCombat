#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 2:00:00
#SBATCH -o runs/%j.out

# Parse command-line arguments
while getopts ":s:" opt; do
  case $opt in
    s)
      seed=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Set default values if arguments are not provided
seed=42
model_type="dt2"
augment_fraction=0.0
filter_dataset=False

# Apptainer images can only be used outside /home. In this example the
# image is located here
cd /proj/berzelius-aiics-real/users/x_denma/projs/ocmdt

# Execute my Apptainer image binding in the current working directory
# containing the Python script I want to execute
cmd="enroot start -e WANDB_API_KEY='f832ecbebaa081e6438201bd475fe26f9f0b1d82' --rw --mount .:/app devcontainer sh -c 'python dt_main.py replay_buffer.filter_dataset=$filter_dataset replay_buffer.augment_fraction=$augment_fraction env.seed=$seed model.model_type=$model_type logger.mode=online'"
eval $cmd