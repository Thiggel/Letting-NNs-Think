. jobs/environment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --finetune_layers 5 \
  --finetune_layers 5 \
  --experiment_name OneLayerRecurrentTransformer_MiddleLayerFineTuning_GPTNeo_125m_Ultrafeedback
