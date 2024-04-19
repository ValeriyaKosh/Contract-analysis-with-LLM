#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=trial1.%J.out
#SBATCH --error=trial1.%J.err
#SBATCH --mail-type=END 
#SBATCH --mail-user=valeriya.koshevaya@aalto.fi

# Choose the model that we want to use
module load model-huggingface/all

module load miniconda
source activate ./LLMTrial1

  # run python
srun python3 myLLM1.py

