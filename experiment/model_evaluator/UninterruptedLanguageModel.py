from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedTokenizer

from experiment.models import DefaultLightningModule
from .CustomInference import CustomInference


class UninterruptedLanguageModel(HFLM):
    def __init__(
        self,
        pretrained: DefaultLightningModule,
        tokenizer: PreTrainedTokenizer,
        *args,
        **kwargs,
    ):
        pretrained = CustomInference(pretrained, tokenizer)

        super().__init__(pretrained, tokenizer, *args, **kwargs)

    def generate_until(self, requests) -> list[str]:
        res = []
        for request in requests:
            context, gen_kwargs = request.arguments

            max_new_tokens = gen_kwargs.get("max_new_tokens", 250)
            decoded = self.custom_inference.generate(
                context, max_new_tokens=max_new_tokens
            )

            print(f"INPUT: {context}, OUTPUT: {decoded}")

            res.append(decoded)
        return res
