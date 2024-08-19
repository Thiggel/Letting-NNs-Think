. jobs/environment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --make_layer_recurrent -1 \
  --finetune_layers 10,11 \
  --num_runs 1 \
  --use_fixed_num_steps \
  --experiment_name OneLayerRecurrentTransformer_FixedThreeSteps_LastLayerFineTuning_GPTNeo_125m_Ultrafeedback
