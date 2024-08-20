. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2b" \
  --finetune_layers 5 \
  --finetune_layers 5 \
  --experiment_name OneLayerRecurrentTransformer_MiddleLayerFineTuning_Gemma_2b_Ultrafeedback
