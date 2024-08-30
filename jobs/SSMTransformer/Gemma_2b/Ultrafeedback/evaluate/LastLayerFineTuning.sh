. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --recurrent_mode "mamba" \
  --finetune_layers -1 \
  --num_runs 1 \
  --checkpoint model_Mamba_DynamicNumSteps.pt \
  --experiment_name Mamba_DynamicNumSteps
