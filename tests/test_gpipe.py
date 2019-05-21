import time

import pytest
import torch
import torch.nn as nn

from torchgpipe.gpipe import GPipe


def test_parameters():
    model = nn.Sequential(nn.Linear(1, 1))
    gpipe = GPipe(model, balance=[1], devices=['cpu'], chunks=1)
    assert list(gpipe.parameters()) != []


def test_non_sequential():
    with pytest.raises(TypeError):
        GPipe(nn.Module(), balance=[1], devices=['cpu'])


@pytest.mark.parametrize('balance', [[2], [1, 1]])
def test_sequential_like(balance):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = GPipe(model, balance, devices=['cpu', 'cpu'])

    assert len(model) == 2
    assert list(model) == [a, b]

    assert model[0] is a
    assert model[1] is b
    with pytest.raises(IndexError):
        _ = model[2]

    assert model[-1] is b
    assert model[-2] is a


def test_balance_wrong_length():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)

    with pytest.raises(ValueError):
        GPipe(model, balance=[1])

    with pytest.raises(ValueError):
        GPipe(model, balance=[3])


def test_too_few_devices():
    x = nn.Linear(1, 1)
    model = nn.Sequential(x, x, x, x)

    with pytest.raises(ValueError):
        # len(balance) > len(devices)
        model = GPipe(model, balance=[1, 1, 1, 1], devices=['cpu'])


def test_identicalness():
    def sum_grad(parameters):
        return sum([p.grad.sum() for p in parameters if p.grad is not None])

    def zero_grad(parameters):
        for p in parameters:
            p.grad = None

    inputs = torch.rand(8, 1)
    model = nn.Sequential(
        nn.Linear(1, 2),
        nn.Linear(2, 4),
        nn.Linear(4, 2),
        nn.Linear(2, 1),
    )

    # Without GPipe
    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    grad_without_gpipe = sum_grad(model.parameters())

    zero_grad(model.parameters())

    # With GPipe
    model = GPipe(model, [2, 2], devices=['cpu', 'cpu'], chunks=4)

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    grad_with_gpipe = sum_grad(model.parameters())

    # Both grads should be identical.
    assert torch.allclose(grad_with_gpipe, grad_without_gpipe)


def test_batch_size_indivisible():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(7, 1))

    # Indivisible batch size is legal.
    assert not record


def test_batch_size_small():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(2, 1))

    # Small batch size is legal.
    assert not record


def test_checkpoint_option():
    def count_grad_fn(grad_fn, name, visited=set()):
        if grad_fn in visited:
            return 0
        visited.add(grad_fn)

        if grad_fn is None:
            return 0
        if grad_fn.__class__.__name__ == name:
            return 1

        counter = 0
        for next_grad_fn, _ in grad_fn.next_functions:
            counter += count_grad_fn(next_grad_fn, name, visited=visited)
        return counter

    model = nn.Sequential(nn.Linear(1, 1))
    input = torch.rand(2, 1)

    always = GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='always')
    except_last = GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='except_last')
    never = GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='never')

    always_output = always(input)
    except_last_output = except_last(input)
    never_output = never(input)

    assert count_grad_fn(always_output.grad_fn, 'CheckpointBackward') == 2
    assert count_grad_fn(except_last_output.grad_fn, 'CheckpointBackward') == 1
    assert count_grad_fn(never_output.grad_fn, 'CheckpointBackward') == 0


def test_checkpoint_option_invalid():
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError):
        GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='INVALID_CHECKPOINT')


def test_checkpoint_eval():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=2)
    input = torch.rand(2, 1)

    def find_grad_fn(grad_fn, name):
        if grad_fn is None:
            return False
        if grad_fn.__class__.__name__ == name:
            return True
        for next_grad_fn, _ in grad_fn.next_functions:
            if find_grad_fn(next_grad_fn, name):
                return True
        return False

    model.train()
    train_output = model(input)
    assert find_grad_fn(train_output.grad_fn, 'CheckpointBackward')
    assert find_grad_fn(train_output.grad_fn, 'RecomputeBackward')

    model.eval()
    eval_output = model(input)
    assert not find_grad_fn(eval_output.grad_fn, 'CheckpointBackward')
    assert not find_grad_fn(eval_output.grad_fn, 'RecomputeBackward')


def test_no_grad():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=2)
    input = torch.rand(2, 1)

    latent = None

    def hook(module, input, outputs):
        _ = module
        _ = input

        output, _ = outputs

        nonlocal latent
        latent = output

    partition = list(model.partitions())[0]
    partition.register_forward_hook(hook)

    with torch.no_grad():
        model(input)

    assert latent.grad_fn is None


def test_exception():
    class ExpectedException(Exception):
        pass

    class Raise(nn.Module):
        def forward(self, *_):
            raise ExpectedException()

    model = nn.Sequential(Raise())
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=1)

    with pytest.raises(ExpectedException):
        model(torch.rand(1))


def test_exception_early_stop():
    class ExpectedException(Exception):
        pass

    class Counter(nn.Module):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def forward(self, x):
            self.counter += 1
            time.sleep(0.01)
            return x

    class Raise(nn.Module):
        def forward(self, x):
            raise ExpectedException()

    count_front = Counter()
    count_back = Counter()
    model = nn.Sequential(count_front, Raise(), count_back)
    model = GPipe(model, balance=[1, 1, 1], devices=['cpu', 'cpu', 'cpu'], chunks=1000)

    with pytest.raises(ExpectedException):
        model(torch.rand(1000, 1))

    # This test is flaky because it relies on different speed among two partitions.
    # But to fail this test, the time to get an exception should be later than
    # 10 seconds (0.01 * 1000.) This situation doesn't seem to happen.
    count_front_counter = count_front.counter
    assert 1 <= count_front_counter < 1000
    assert count_back.counter == 0

    # The first partition should be already stopped.
    time.sleep(0.1)
    assert count_front.counter == count_front_counter


def test_input_pair():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_a = nn.Linear(1, 1)
            self.fc_b = nn.Linear(1, 1)

        def forward(self, a_and_b):
            a, b = a_and_b
            return (self.fc_a(a), self.fc_b(b))

    model = nn.Sequential(Two())
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=2)

    a = torch.rand(10, 1, requires_grad=True)
    b = torch.rand(10, 1, requires_grad=True)

    a_out, b_out = model((a, b))
    loss = (a_out + b_out).mean()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None


def test_input_singleton():
    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, only_a):
            a, = only_a
            return (self.fc(a),)

    model = nn.Sequential(One())
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=2)

    a = torch.rand(10, 1, requires_grad=True)

    a_out, = model((a,))
    loss = a_out.mean()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
    assert a.grad is not None


def test_input_varargs():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'])

    a = torch.rand(1)
    b = torch.rand(1)

    # TypeError: forward() takes 2 positional arguments but 3 were given
    with pytest.raises(TypeError):
        model(a, b)


def test_non_tensor():
    class NonTensor(nn.Module):
        def forward(self, _):
            return 'hello'

    model = nn.Sequential(NonTensor())
    model = GPipe(model, balance=[1], devices=['cpu'])
    x = torch.rand(1)

    # TypeError: expected Tensor as element 0 in argument 0, but got str
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model('hello')


def test_non_tensor_tuple():
    class NonTensorTuple(nn.Module):
        def forward(self, x):
            return (x, 'hello')

    model = nn.Sequential(NonTensorTuple())
    model = GPipe(model, balance=[1], devices=['cpu'])
    x = torch.rand(1)

    # TypeError: CheckpointBackward.forward: expected Variable (got str) for return value 1
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model((x, 'hello'))


def test_lockstep():
    timeline = []

    class DelayedLog(nn.Module):
        def __init__(self, i, seconds):
            super().__init__()
            self.i = i
            self.j = 0
            self.seconds = seconds

        def forward(self, x):
            time.sleep(self.seconds)

            timeline.append((self.i, self.j))
            self.j += 1

            return x

    model = nn.Sequential(DelayedLog(0, seconds=0), DelayedLog(1, seconds=0.1))
    model = GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=3)

    x = torch.rand(3, 1)
    model(x)

    # Expected timeline: (Logs are recorded at !)
    #
    # Partition #0: 0! 1!   2!
    # Partition #1:    000! 111! 222!
    #
    assert timeline == [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]


@pytest.mark.parametrize('checkpoint', ['never', 'always', 'except_last'])
def test_deferred_batch_norm(checkpoint):
    bn = nn.BatchNorm2d(3)
    bn_under_gpipe = nn.BatchNorm2d(3)

    gpipe = GPipe(nn.Sequential(bn_under_gpipe), balance=[1], devices=['cpu'], chunks=2,
                  checkpoint=checkpoint, deferred_batch_norm=True)

    x = torch.rand(4, 3, 10, 10)
    gpipe(x).mean().backward()
    bn(x)

    assert torch.allclose(bn_under_gpipe.running_mean, bn.running_mean, atol=1e-4)
    assert torch.allclose(bn_under_gpipe.running_var, bn.running_var, atol=1e-4)
