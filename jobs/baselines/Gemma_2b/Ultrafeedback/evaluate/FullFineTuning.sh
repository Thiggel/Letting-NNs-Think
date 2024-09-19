. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --finetune_layers "all" \
  --checkpoint model_Baseline_FullFineTuning.pt \
  --experiment_name Baseline_FullFineTuning
