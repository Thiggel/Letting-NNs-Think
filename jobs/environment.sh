module purge
module load 2023
module load Anaconda3/2023.07-2
module load GCC/12.3.0
module load CUDA/12.1.1

export CUBLAS_WORKSPACE_CONFIG=:4096:8

conda activate letting_nns_think

pip install -r requirements.txt
