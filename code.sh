#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --cpus_per_task=4
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --output=trial1.%J.out
#SBATCH --error=trial1.%J.err

# Choose the model that we want to use
module load model-llama2/13b-chat

????????? # Choose the llama.cpp model quantization we want to use
????????? module load model-llama.cpp/q4_1-2023-08-28

# Get the path to model weights
echo $MODEL_WEIGHTS

  # run python
srun python LLM-pdf-reader/code.py





#run on Login node:
 # setup environment
module load miniconda 
conda env create -f LLM-pdf-reader/env.yml -p ./LLMTrial1 #(only once)
source activate ./LLMTrial1

sbatch code.sh
