. jobs/environment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --make_layer_recurrent -1 \
  --use_ssm \
  --finetune_layers -1 \
  --experiment_name SSMTransformer_LastLayerFineTuning_GPTNeo_125m_Ultrafeedback
