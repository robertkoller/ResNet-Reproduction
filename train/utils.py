import random, numpy, torch

# check what device we can use
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# set seed so that we get deterministic results and can compare different versions
def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Set seed: {seed}")
    