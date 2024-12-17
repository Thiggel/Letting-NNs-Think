import torch.nn as nn
import inspect


class SequentialTransformerLayer(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        sig = inspect.signature(self.layers[0].forward)
        supported_kwargs = {
            key: value for key, value in kwargs.items() if key in sig.parameters
        }
        rest = None
        for layer in self.layers:
            outputs = layer(x, *args, **supported_kwargs)
            if type(x) == tuple:
                x = outputs[0]

        print("rest", outputs)
        exit()

        return x, rest

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def append(self, layer):
        self.layers.append(layer)

    def extend(self, layers):
        self.layers.extend(layers)

    def insert(self, index, layer):
        self.layers.insert(index, layer)
