. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --recurrent_mode "mamba" \
  --finetune_layers -1 \
  --num_runs 1 \
  --use_fixed_num_steps \
  --checkpoint model_Mamba_InitialStateProj_ThreeSteps.pt \
  --experiment_name Mamba_InitialStateProj_ThreeSteps
