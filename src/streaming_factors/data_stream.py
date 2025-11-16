import numpy as np

def simulate_returns(num_assets: int = 5, num_steps: int = 500, seed: int = 0):
    """
    Toy return generator for early testing.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, size=(num_steps, num_assets))
    return returns


def stream_returns(returns):
    """
    Yield one return vector at a time
    """
    for r in returns:
        yield r