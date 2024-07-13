. jobs/enviroment.sh

python -m experiment \
  --model_name "EleutherAI/gpt-neo-125m" \
  --make_layer_recurrent -1 \
  --experiment_name OneLayerRecurrentTransformer_FullFineTuning_GPTNeo_125m_Ultrafeedback
