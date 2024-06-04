#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 05:00:00
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


# Apptainer images can only be used outside /home. In this example the
# image is located here
cd /proj/berzelius-aiics-real/users/x_denma/

apptainer exec --env "WANDB_API_KEY=f832ecbebaa081e6438201bd475fe26f9f0b1d82" --nv -B ./projs/CloseAirCombat:/app berzdev_latest.sif 'cd /app & python -m a2c_bandit_ce.a2c_mujoco'
