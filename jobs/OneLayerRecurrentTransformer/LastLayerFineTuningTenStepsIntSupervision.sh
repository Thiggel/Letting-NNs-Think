. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --finetune_layers -1 \
  --num_runs 1 \
  --use_fixed_num_steps \
  --num_steps 10 \
  --no_evaluate \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --max_epochs 10 \
  --time_embedding \
  --experiment_name RecurrentTransformer_TenSteps_TimeEmbedding_IntSupervision
