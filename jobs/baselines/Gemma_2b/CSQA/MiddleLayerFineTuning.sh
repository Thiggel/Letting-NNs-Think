. jobs/enviroment.sh

python -m experiment \
  --model_name "google/gemma-2b" \
  --finetune_layers 5 \
  --experiment_name Baseline_MiddleLayerFineTuning_Gemma_2b_CSQA
