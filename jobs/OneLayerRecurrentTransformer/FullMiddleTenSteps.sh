. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent 12 \
  --finetune_layers "all" \
  --num_runs 1 \
  --use_fixed_num_steps \
  --no_evaluate \
  --num_steps 10 \
  --max_epochs 30 \
  --experiment_name FullMiddleTenSteps
