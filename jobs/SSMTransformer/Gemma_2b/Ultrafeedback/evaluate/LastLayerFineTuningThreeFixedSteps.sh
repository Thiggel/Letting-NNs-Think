. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --recurrent_mode "mamba" \
  --finetune_layers -1 \
  --num_runs 1 \
  --use_fixed_num_steps \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --checkpoint model_OneLayerRecurrentTransformer_LastLayerFineTuning_Gemma_2b_Ultrafeedback.pt \
  --experiment_name SSMTransformer_ThreeFixedSteps_LastLayerFineTuning_Gemma_2b_Ultrafeedback
