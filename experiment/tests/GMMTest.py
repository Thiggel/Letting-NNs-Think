import torch

from experiment.models.uninterrupted_language_model.GMMHead import GMMHead


def test_gradient_flow():
    # Create a minimal test case
    hidden_size = 4
    batch_size = 2
    seq_len = 3
    gmm = GMMHead(hidden_size, n_components=2)

    # Create initial hidden states that require grad
    initial_hidden = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    hidden = initial_hidden

    # Store all intermediate states
    all_states = [hidden]

    # Do 5 steps of sampling
    for step in range(5):
        sampled = gmm.reparameterized_sample(hidden)
        hidden = sampled
        all_states.append(hidden)

    # Create a loss based on final hidden state
    final_loss = all_states[-1].sum()

    # Compute gradients
    final_loss.backward()

    # Check if gradients flowed back
    initial_grad = initial_hidden.grad

    print(f"Initial hidden state gradient exists: {initial_grad is not None}")
    print(f"Initial hidden state gradient norm: {initial_grad.norm().item()}")
    print("\nGradient at each parameter in first mixture weight layer:")
    for name, param in gmm.mixture_weights_head.named_parameters():
        print(f"{name} gradient norm: {param.grad.norm().item()}")

    return all(p.grad is not None for p in gmm.parameters())


# Run test
test_gradient_flow()


def test_computation_graph():
    # Create minimal test setup
    hidden_size = 4
    batch_size = 2
    seq_len = 3
    gmm = GMMHead(hidden_size, n_components=2)

    # Create initial hidden state that requires grad
    hidden = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

    # Store all intermediate hidden states
    all_hiddens = []

    # Do 5 steps of sampling
    current = hidden
    for step in range(5):
        # Store current hidden state
        all_hiddens.append(current)

        # Sample next state
        current = gmm.reparameterized_sample(current)
        # Call retain_grad() on intermediate state
        current.retain_grad()

    # Add final state
    all_hiddens.append(current)
    current.retain_grad()

    # Create loss based on final hidden state
    final_loss = all_hiddens[-1].sum()

    # Backward pass
    final_loss.backward()

    # Check gradients for each hidden state
    print("\nChecking gradients after backward:")
    print("-" * 40)
    for i, hidden_state in enumerate(all_hiddens):
        has_grad = hidden_state.grad is not None
        requires_grad = hidden_state.requires_grad
        is_leaf = hidden_state.is_leaf
        grad_fn_name = (
            type(hidden_state.grad_fn).__name__ if hidden_state.grad_fn else "None"
        )

        print(f"\nStep {i}:")
        print(f"  Requires grad: {requires_grad}")
        print(f"  Has grad: {has_grad}")
        print(f"  Is leaf: {is_leaf}")
        print(f"  Grad function: {grad_fn_name}")
        if hidden_state.grad is not None:
            print(f"  Gradient norm: {hidden_state.grad.norm().item():.6f}")

    print("\nVerifying all states got gradients:")
    all_have_grads = all(h.grad is not None for h in all_hiddens)
    print(f"All states have gradients: {'✓' if all_have_grads else '✗'}")

    return all_have_grads


# Run test
test_computation_graph()
