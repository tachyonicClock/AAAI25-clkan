import torch
from kan import KAN as PyKAN

from clkan.plot.kan import pykan_activations


def test_pykan_activations():
    num_samples = 10
    model = PyKAN([2, 3, 2])
    x = torch.randn(num_samples, 2)

    activations = pykan_activations(model, x)

    assert len(activations) == 3
    assert activations[0].shape == (num_samples, 2)
    assert activations[1].shape == (num_samples, 3)
    assert activations[2].shape == (num_samples, 2)
