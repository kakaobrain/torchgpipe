# torchgpipe

[![Build Status](https://travis-ci.org/kakaobrain/torchgpipe.svg?branch=master)](https://travis-ci.org/kakaobrain/torchgpipe)

A [GPipe](https://arxiv.org/abs/1811.06965) implementation in PyTorch.

```python
from torchgpipe import GPipe

model = nn.Sequential(a, b, c, d)
model = GPipe(model, balance=[1,1,1,1], chunks=8)

for input in data_loader:
    outputs = model(input)
```

---

This project is still under development. Any public API would be changed
without deprecation warnings until v0.1.0.
