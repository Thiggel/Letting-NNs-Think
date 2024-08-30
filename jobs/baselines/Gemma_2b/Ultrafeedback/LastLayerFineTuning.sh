. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --finetune_layers -1 \
  --no_evaluate \
  --experiment_name Baseline_LastLayerFineTuning
