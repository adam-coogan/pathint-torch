# pathint\_torch

[![docs](https://readthedocs.org/projects/pathint-torch/badge/?version=latest)](http://pathint-torch.readthedocs.io/?badge=latest)

`pathint_torch` is a PyTorch implementation of the [path integral sampler](https://arxiv.org/abs/2111.15141),
a method based on the Schrödinger bridge problem for sampling from (unnormalized)
probability densities. Behind the scenes it relies on [torchsde](https://github.com/google-research/torchsde)
to solve stochastic differential equations.

Check out the [docs](https://pathint-torch.readthedocs.io/en/latest/) for more details,
or try out the scripts applying the method to some low-dimensional problems (runnable
on a laptop).

## Installation

Running
```
git clone git@github.com:adam-coogan/pathint-torch.git
cd pathint-torch
pip install .
```
will install the `pathint_torch` package.
