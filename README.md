## Project Structure

```
├── experiment                          - all the code is in here
│   ├── LanguageDataModule.py           - all the dataset loading and creation of next-token prediction labels happens here
│   ├── LMLightningModule.py            - all the model logic including making layers recurrent etc. is in here
│   ├── __main__.py
│   ├── RecurrentTransformerLayer.py    - this module is used my LMLightningModule for recurrent Transformer layers
│   └── utils
│       ├── accuracy.py                 - Next-token accuracy function
│       ├── add_pad_token.py
│       ├── args.py                     - Command line arguments class (so that my python linter can infer the cmd arg types)
│       ├── get_num_workers.py
│       ├── get_training_args.py        - All command line args are parsed with this function
│       ├── print_mean_std.py
│       ├── run_different_seeds.py
│       ├── run.py                      - A distinct run of the application is initiated with this function
│       └── set_seed.py
```

## How to run the experiment

All available Snellius jobs are in the directory `jobs`. 

The experiment can be invoked like this:

```
usage: python -m experiment [-h] [--seeds SEEDS] [--num_runs NUM_RUNS] [--model_name MODEL_NAME]
                   [--finetune_layers FINETUNE_LAYERS] [--remove_layers REMOVE_LAYERS]
                   [--make_layer_recurrent MAKE_LAYER_RECURRENT]
                   [--dataset {ultrafeedback,csqa_full,arc_full,piqa_full,siqa_full,openhermes,alpaca,gsm8k}]
                   [--seq_length SEQ_LENGTH] [--train_batch_size TRAIN_BATCH_SIZE]
                   [--eval_batch_size EVAL_BATCH_SIZE] [--no_logger]
                   [--experiment_name EXPERIMENT_NAME] [--max_epochs MAX_EPOCHS]
                   [--warmup_steps WARMUP_STEPS]

Training arguments

options:
  -h, --help            show this help message and exit
  --seeds SEEDS         Random seeds
  --num_runs NUM_RUNS   The number of runs
  --model_name MODEL_NAME
                        The model name to be used
  --finetune_layers FINETUNE_LAYERS
                        The layers to fine-tune
  --remove_layers REMOVE_LAYERS
                        The layers to remove
  --make_layer_recurrent MAKE_LAYER_RECURRENT
                        The layer to make recurrent
  --dataset {ultrafeedback,csqa_full,arc_full,piqa_full,siqa_full,openhermes,alpaca,gsm8k}
                        The dataset to use for training
  --seq_length SEQ_LENGTH
                        The maximum sequence length
  --train_batch_size TRAIN_BATCH_SIZE
                        The training batch size
  --eval_batch_size EVAL_BATCH_SIZE
                        The evaluation batch size
  --no_logger           Whether to use a logger
  --experiment_name EXPERIMENT_NAME
                        The name of the experiment
  --max_epochs MAX_EPOCHS
                        The maximum number of epochs
  --warmup_steps WARMUP_STEPS
                        The number of warmup steps
```
