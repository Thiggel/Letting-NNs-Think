import torch

checkpoint = torch.load(
    "/projects/prjs1017/LettingLMsThinkSynthetic/Pythia_Arithmetic_nGPT_10Step_TimeEmbedding_1.pt",
    map_location=torch.device("cpu"),
)

print(checkpoint.keys())

for param_name, param in checkpoint.items():
    if "eigen_rate" in param_name:

        print(param_name, param.mean(), param.max())
