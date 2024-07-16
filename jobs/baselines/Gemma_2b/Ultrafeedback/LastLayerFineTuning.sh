. jobs/environment.sh

python -m experiment \
  --model_name "google/gemma-2b" \
  --finetune_layers -1 \
  --max_epochs 10 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --experiment_name Baseline_LastLayerFineTuning_Gemma_2b_Ultrafeedback
