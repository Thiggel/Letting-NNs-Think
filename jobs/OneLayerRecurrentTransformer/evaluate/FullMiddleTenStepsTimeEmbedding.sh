. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent 12 \
  --finetune_layers "all" \
  --num_runs 1 \
  --use_fixed_num_steps \
  --num_steps 10 \
  --max_epochs 30 \
  --time_embedding \
  --checkpoint model_FullMiddleTenStepsTimeEmbedding.pt \
  --experiment_name FullMiddleTenStepsTimeEmbedding
