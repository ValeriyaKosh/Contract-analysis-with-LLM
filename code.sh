#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --cpus_per_task=4
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --output=trial1.%J.out
#SBATCH --error=trial1.%J.err

# Choose the model that we want to use
module load model-huggingface/all

  # run python
srun python Contract-analysis-with-LLM/code.py
