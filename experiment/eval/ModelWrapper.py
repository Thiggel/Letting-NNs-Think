import torch
from lm_eval.api.model import LM
from tqdm import tqdm


class ModelWrapper(LM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        res = []
        for request in tqdm(requests, desc="loglikelihood"):
            context, continuation = request.arguments
            inputs = self.tokenizer(context + continuation, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_likelihood = -outputs.loss.item() * inputs["input_ids"].size(1)
            is_greedy = continuation in self.greedy_until([(context, "")])
            res.append((log_likelihood, is_greedy))
        return res

    def loglikelihood_rolling(self, requests) -> list[float]:
        res = []
        for request in tqdm(requests, desc="loglikelihood_rolling"):
            context = request.arguments[0]
            inputs = self.tokenizer(context, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_likelihood = -outputs.loss.item() * inputs["input_ids"].size(1)
            res.append(log_likelihood)
        return res

    def generate_until(self, requests) -> list[str]:
        res = []
        for request in tqdm(requests, desc="generate_until"):
            context, gen_kwargs = request.arguments
            input_ids = self.tokenizer.encode(context, return_tensors="pt")
            output = self.model.generate(input_ids, **gen_kwargs)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            res.append(decoded)
        return res

    def greedy_until(self, requests):
        res = []
        for context, _ in tqdm(requests, desc="greedy_until"):
            input_ids = self.tokenizer.encode(context, return_tensors="pt")
            output = self.model.generate(input_ids, max_length=100)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            res.append(decoded)
        return res
