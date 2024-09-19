. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent 12 \
  --finetune_layers 12 \
  --num_runs 1 \
  --use_random_num_steps \
  --no_evaluate \
  --max_epochs 50 \
  --experiment_name RecurrentTransformer_RandomNumSteps_50_Epochs_MiddleLayer
