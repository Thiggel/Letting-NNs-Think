. jobs/environment.sh

python -m experiment \
  --model_name "google/gemma-2b" \
  --make_layer_recurrent -1 \
  --finetune_layers -1 \
  --experiment_name OneLayerRecurrentTransformer_LastLayerFineTuning_Gemma_2b_Ultrafeedback
