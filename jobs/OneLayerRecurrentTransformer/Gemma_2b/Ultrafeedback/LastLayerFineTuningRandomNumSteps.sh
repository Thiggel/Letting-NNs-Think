. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --finetune_layers -1 \
  --num_runs 1 \
  --use_random_num_steps \
  --experiment_name OneLayerRecurrentTransformer_RandomNumSteps_LastLayerFineTuning_Gemma_2b_Ultrafeedback
