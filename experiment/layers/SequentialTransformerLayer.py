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
        for layer in self.layers:
            x = layer(x, *args, **supported_kwargs)
            print("xxx", x)
            if type(x) == tuple:
                x = x[0]
                rest = x[1:]

        print("rest", rest)
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
