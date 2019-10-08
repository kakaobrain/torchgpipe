import pytest
import torch


@pytest.fixture(autouse=True)
def manual_seed_zero():
    torch.manual_seed(0)


@pytest.fixture(scope='session')
def cuda_sleep():
    # From test/test_cuda.py in PyTorch.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(1000000)
    end.record()
    end.synchronize()
    cycles_per_ms = 1000000 / start.elapsed_time(end)

    def cuda_sleep(seconds):
        torch.cuda._sleep(int(seconds * cycles_per_ms * 1000))
    return cuda_sleep


def pytest_report_header():
    return f'torch: {torch.__version__}'
