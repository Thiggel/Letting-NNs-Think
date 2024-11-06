module purge
module load cuda12.1/toolkit/12.1

export SCRATCH_LOCAL="/var/scratch/flaitenb/"

cd $SCRATCH_LOCAL
mkdir flaitenberger

cd $HOME/Letting-NNs-Think

# Base directory
export BASE_CACHE_DIR="$SCRATCH_LOCAL/flaitenberger"

# Hugging Face
export HF_HOME="$BASE_CACHE_DIR"
export HF_DATASETS_CACHE="$BASE_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$BASE_CACHE_DIR/transformers"
export HF_MODULES_CACHE="$BASE_CACHE_DIR/modules"

# DeepSpeed
export DEEPSPEED_CACHE_DIR="$BASE_CACHE_DIR/deepspeed"

# Weights & Biases
export WANDB_DIR="$BASE_CACHE_DIR/wandb"

# PyTorch Lightning
# Note: PyTorch Lightning doesn't use an environment variable, 
# but you can use this in your Python code
export PYTORCH_LIGHTNING_HOME="$BASE_CACHE_DIR/lightning_logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

activate letting_nns_think
conda activate letting_nns_think

pip install -r requirements.txt
