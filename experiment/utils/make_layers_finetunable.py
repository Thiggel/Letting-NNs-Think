from transformers import AutoModelForCausalLM


def make_layers_finetunable(model: AutoModelForCausalLM, finetune_layers: list[int]):
    if finetune_layers != "all":
        for param in model.parameters():
            param.requires_grad = False

        for i in finetune_layers:
            for param in model.transformer.h[i].parameters():
                param.requires_grad = True
