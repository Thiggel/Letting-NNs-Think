. jobs/environment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --make_layer_recurrent -1 \
  --recurrent_mode 'adaptive_transformer' \
  --finetune_layers -1 \
  --experiment_name AdaptiveTransformer_LastLayerFineTuning_GPTNeo_125m_Ultrafeedback
