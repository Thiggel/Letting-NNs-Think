. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent 12 \
  --finetune_layers "all" \
  --num_runs 1 \
  --use_fixed_num_steps \
  --num_steps 3 \
  --checkpoint model_RecurrentTransformer_ThreeSteps_50_Epochs_MiddleLayerRecurrentFullFineTuning.pt \
  --experiment_name RecurrentTransformer_ThreeSteps_50_Epochs_MiddleLayerRecurrentFullFineTuning
