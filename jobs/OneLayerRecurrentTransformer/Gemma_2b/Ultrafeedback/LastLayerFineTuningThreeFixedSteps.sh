. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent 25 \
  --finetune_layers 24,25 \
  --num_runs 1 \
  --no_logger \
  --use_fixed_num_steps \
  --experiment_name OneLayerRecurrentTransformer_ThreeFixedSteps_LastLayerFineTuning_Gemma_2b_Ultrafeedback
