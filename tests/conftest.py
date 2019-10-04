import pytest
import torch


@pytest.fixture(autouse=True)
def manual_seed_zero():
    torch.manual_seed(0)


def pytest_report_header():
    return f'torch: {torch.__version__}'
