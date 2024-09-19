. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent 12 \
  --finetune_layers 12 \
  --num_runs 1 \
  --use_fixed_num_steps \
  --checkpoint model_RecurrentTransformer_ThreeSteps_50_Epochs_MiddleLayer.pt \
  --experiment_name RecurrentTransformer_ThreeSteps_50_Epochs_MiddleLayer
