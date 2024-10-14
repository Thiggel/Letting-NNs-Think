from typing import Optional, Tuple, Union


class Args:
    model_name: str
    dataset: str
    seq_length: int
    train_batch_size: int
    eval_batch_size: int
    seeds: list[int]
    num_runs: int
    finetune_layers: Optional[list[int]]
    make_layers_recurrent: Union[int, Tuple[int, int]]
    recurrent_mode: str
    num_steps: int
    use_skip_connection: bool
    use_fixed_num_steps: bool
    use_random_num_steps: bool
    logger: bool
    experiment_name: str
    max_epochs: int
    warmup_steps: int
    checkpoint: Optional[str]
    load_from_checkpoint: Optional[str]
    evaluate: bool
    use_time_embedding: bool
    use_gating: bool
    use_reinforce: bool
    gamma: float
    temperature: float
    max_grad_norm: float
    baseline_decay: float
    use_random_intermediate_supervision: bool
    evaluation_metrics: list[str]
