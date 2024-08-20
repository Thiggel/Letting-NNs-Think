. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2b" \
  --make_layer_recurrent -1 \
  --experiment_name OneLayerRecurrentTransformer_FullFineTuning_Gemma_2b_Ultrafeedback
