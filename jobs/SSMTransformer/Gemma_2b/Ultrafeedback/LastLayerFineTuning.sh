. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --recurrent_mode "ssm" \
  --finetune_layers -1 \
  --num_runs 1 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --experiment_name SSMTransformer_LastLayerFineTuning_Gemma_2b_Ultrafeedback
