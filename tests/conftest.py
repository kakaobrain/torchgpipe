import pytest
import torch


@pytest.fixture(autouse=True)
def manual_seed_zero():
    torch.manual_seed(0)
