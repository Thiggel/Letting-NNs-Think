. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --finetune_layers -1 \
  --num_runs 1 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --no_evaluate \
  --experiment_name RecurrentTransformer_DynamicNumSteps
