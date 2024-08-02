. jobs/environment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --make_layer_recurrent -1 \
  --recurrent_mode 'adaptive_transformer' \
  --finetune_layers -1 \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --experiment_name RLAdaptiveTransformer_LastLayerFineTuning_GPTNeo_125m_Ultrafeedback
