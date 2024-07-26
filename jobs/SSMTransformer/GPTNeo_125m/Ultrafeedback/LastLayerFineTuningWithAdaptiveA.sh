. jobs/environment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --make_layer_recurrent -1 \
  --use_ssm \
  --use_adaptive_A \
  --finetune_layers -1 \
  --experiment_name SSMTransformerWithAdaptiveA_LastLayerFineTuning_GPTNeo_125m_Ultrafeedback
