"""
torchgpipe
==========

A GPipe_ implementation in PyTorch_.

.. _GPipe: https://arxiv.org/abs/1811.06965
.. _PyTorch: https://pytorch.org/

.. sourcecode:: python

   from torchgpipe import GPipe

   model = nn.Sequential(a, b, c, d)
   model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)

   for input in data_loader:
       output = model(input)

What is GPipe?
~~~~~~~~~~~~~~

GPipe is a scalable pipeline parallelism library published by Google Brain,
which allows efficient training of large, memory-consuming models. According to
the paper, GPipe can train a 25x larger model by using 8x devices (TPU), and
train a model 3.5x faster by using 4x devices.

`GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
<https://arxiv.org/abs/1811.06965>`_

Google trained AmoebaNet-B with 557M parameters over GPipe. This model has
achieved 84.3% top-1 and 97.0% top-5 accuracy on ImageNet classification
benchmark (the state-of-the-art performance as of May 2019).

Links
~~~~~

- Source Code: https://github.com/kakaobrain/torchgpipe
- Documentation: https://torchgpipe.readthedocs.io/
- Original Paper: https://arxiv.org/abs/1811.06965

"""
from setuptools import setup


about = {}  # type: ignore
with open('torchgpipe/__version__.py') as f:
    exec(f.read(), about)  # pylint: disable=W0122
version = about['__version__']
del about


setup(
    name='torchgpipe',

    version=version,

    license='Apache License 2.0',
    url='https://github.com/kakaobrain/torchgpipe',
    author='Kakao Brain',
    maintainer='Heungsub Lee, Myungryong Jeong, Chiheon Kim',

    description='GPipe for PyTorch',
    long_description=__doc__,
    keywords='pytorch gpipe',

    zip_safe=False,

    packages=['torchgpipe', 'torchgpipe.balancing'],
    package_data={'torchgpipe': ['py.typed']},
    py_modules=['torchgpipe_balancing'],

    install_requires=['torch>=1.1'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=4'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Typing :: Typed',
    ],
)
