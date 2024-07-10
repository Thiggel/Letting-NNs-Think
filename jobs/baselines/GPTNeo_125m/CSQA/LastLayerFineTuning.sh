. jobs/enviroment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --finetune_layers -1 \
  --experiment_name Baseline_LastLayerFineTuning_GPTNeo_125m_CSQA
