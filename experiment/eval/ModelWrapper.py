import torch
from lm_eval.api.model import LM
from tqdm import tqdm


class ModelWrapper(LM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def _to_device(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        res = []
        for request in tqdm(requests, desc="loglikelihood"):
            context, continuation = request.arguments
            inputs = self._to_device(self.tokenizer(context + continuation, return_tensors="pt"))
            with torch.no_grad():
                print(self.model.device, inputs["input_ids"].device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])

            decoded = self.tokenizer.decode(
                outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True
            )

            print(
                f"Context: {context}\nContinuation: {continuation}\nDecoded: {decoded}\n\n"
            )

            log_likelihood = -outputs.loss.item() * inputs["input_ids"].size(1)
            is_greedy = continuation in self.greedy_until([(context, "")])
            res.append((log_likelihood, is_greedy))
        return res

    def loglikelihood_rolling(self, requests) -> list[float]:
        res = []
        for request in tqdm(requests, desc="loglikelihood_rolling"):
            context = request.arguments[0]
            inputs = self._to_device(self.tokenizer(context, return_tensors="pt"))
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])

            decoded = self.tokenizer.decode(
                outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True
            )

            print(f"Context: {context}\nDecoded: {decoded}\n\n")

            log_likelihood = -outputs.loss.item() * inputs["input_ids"].size(1)
            res.append(log_likelihood)
        return res

    def generate_until(self, requests) -> list[str]:
        res = []
        for request in requests:
            context, gen_kwargs = request.arguments
            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
            if 'until' in gen_kwargs:
                gen_kwargs.pop('until')

            gen_kwargs['max_length'] = 99
            gen_kwargs['max_new_tokens'] = 50
            output = self.model.generate(input_ids, **gen_kwargs)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"Context: {context}\nDecoded: {decoded}\n\n")
            res.append(decoded)
        return res

    def greedy_until(self, requests):
        res = []
        for context, _ in requests:
            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
            output = self.model.generate(input_ids, max_length=100)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"Context: {context}\nDecoded: {decoded}\n\n")
            res.append(decoded)
        return res
