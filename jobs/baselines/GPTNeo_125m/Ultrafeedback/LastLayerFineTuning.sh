. jobs/environment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --finetune_layers 10,11 \
  --experiment_name Baseline_LastLayerFineTuning_GPTNeo_125m_Ultrafeedback
