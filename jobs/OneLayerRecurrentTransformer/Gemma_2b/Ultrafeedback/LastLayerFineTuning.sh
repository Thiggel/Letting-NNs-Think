. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent 25 \
  --finetune_layers 24,25 \
  --num_runs 1 \
  --use_fixed_num_steps \
  --no_logger \
  --experiment_name OneLayerRecurrentTransformer_LastLayerFineTuning_Gemma_2b_Ultrafeedback
