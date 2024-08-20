. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --finetune_layers 24,25 \
  --experiment_name Baseline_LastLayerFineTuning_Gemma_2b_Ultrafeedback
