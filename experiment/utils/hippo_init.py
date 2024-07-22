import torch


def hippo_init(A: torch.nn.Linear, B: torch.nn.Linear, method="LagT"):
    if method == "LagT":
        A.weight.data = torch.tril(torch.ones_like(A.weight.data))
        B.weight.data = torch.ones_like(B.weight.data)

    else:
        N = A.weight.shape[0]

        q = torch.arange(N, dtype=torch.float64)
        col, row = torch.meshgrid(q, q)

        r = 2 * q + 1
        M = -(torch.where(row >= col, r, 0) - torch.diag(q))
        T = torch.sqrt(torch.diag(2 * q + 1))

        with torch.no_grad():
            A.weight.data = T @ M @ torch.linalg.inv(T)

            B.weight.data = torch.ones_like(B.weight.data)
