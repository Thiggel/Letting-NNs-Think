. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --experiment_name Baseline_NoFineTuning
