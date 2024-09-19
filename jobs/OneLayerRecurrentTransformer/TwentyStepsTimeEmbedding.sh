. jobs/environment.sh

srun python -m experiment \
  --model_name "google/gemma-2-2b" \
  --make_layer_recurrent -1 \
  --finetune_layers -1 \
  --num_runs 1 \
  --use_fixed_num_steps \
  --num_steps 20 \
  --no_evaluate \
  --time_embedding \
  --experiment_name RecurrentTransformer_TwentyStepsTimeEmbedding
