. jobs/enviroment.sh

python -m experiment \
  --model_name "google/gemma-2b" \
  --finetune_layers -1 \
  --experiment_name Baseline_LastLayerFineTuning_Gemma_2b_CSQA
