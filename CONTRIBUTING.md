# Contributing to torchgpipe

torchgpipe is currently under development. We will try to make it stable and
usable. Until v0.1.0, which will be our first production release, we don't open
up receiving issues or pull requests. Please wait for a while.

Kakao Brain, the project owner, has authority to update this guide.

## Boundaries

torchgpipe is a library, not an experiment.

The `torchgpipe` Python module, which is the `torchgpipe` folder in this
respository, has responsibility to provide GPipe for CUDA in PyTorch, and
options with some trade-offs which users have freedom to choose. It would not
accept any requirement beyond the original GPipe implementation.

The "torchgpipe" project, which is this repository itself, has responsibility
to make GPipe easy-to-use by deep learning researchers or engineers. It
provides handy resources and documentation for the best practice.

Delicate reproduction of experiments in GPipe paper is out of the
responsibility of this project.

After we release v0.1.0, if your pull request is accepted, we will merge it by
squashing regardless of the work history. Your forked repository should keep
the history.

## Styleguides

- Think of readability, consistency, simplicity, and cohesion.
- Don't put spaces around an operator if it is easier to read
  (`2*i + 1` not `2 * i + 1`.)
- Lint by mypy and Flake8 with our `setup.cfg`.
- Format code by autopep8 and isort with our `setup.cfg`.
- Prefer PyTorch's coding style rather than PEP 8 (`input` is a good name for
  input tensors.)

## Development

### Unit Testing

To run unit tests, you can simply run `python setup.py test`. But if you want
to use advanced testing options, run pytest manually:

```sh
$ pip install pytest
$ pytest
```

For example, you can filter tests by name:

```sh
$ pytest -k 'test_gpipe'
```

### Code Quality

We use mypy and Flake8 to check code quality:

```sh
$ pip install mypy flake8
$ mypy .
$ flake8 torchgpipe tests setup.py
```

We highly recommend to use autopep8 and isort to follow the coding style
automatically:

```sh
$ pip install autopep8 isort
$ autopep8 -ir torchgpipe tests setup.py
$ isort -rc torchgpipe tests setup.py
```
