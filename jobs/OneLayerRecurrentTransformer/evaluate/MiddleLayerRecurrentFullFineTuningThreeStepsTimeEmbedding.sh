. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent 12 \
  --finetune_layers "all" \
  --num_runs 1 \
  --use_fixed_num_steps \
  --time_embedding \
  --checkpoint model_ThreeSteps_MiddleLayerRecurrent_FullFineTuning_TimeEmbedding.pt \
  --experiment_name ThreeSteps_MiddleLayerRecurrent_FullFineTuning_TimeEmbedding
