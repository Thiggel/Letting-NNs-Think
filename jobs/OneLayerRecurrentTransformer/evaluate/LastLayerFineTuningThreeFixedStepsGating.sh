. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --finetune_layers -1 \
  --num_runs 1 \
  --use_fixed_num_steps \
  --gating \
  --checkpoint model_RecurrentTransformer_ThreeSteps_Gating.pt \
  --experiment_name RecurrentTransformer_ThreeSteps_Gating
