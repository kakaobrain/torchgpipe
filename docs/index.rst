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

Documentations
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   gpipe
   guide
   api
   benchmarks
   changelog

Authors and Licensing
~~~~~~~~~~~~~~~~~~~~~

This project is developed by `Heungsub Lee`_, `Myungryong Jeong`_, and `Chiheon
Kim`_ at `Kakao Brain`_, with `Sungbin Lim`_, `Ildoo Kim`_, `Woonhyuk Baek`_,
and `Boogeon Yoon`_'s help. It is distributed under the 3-clause BSD license.

.. _Heungsub Lee: https://subl.ee/
.. _Myungryong Jeong: https://github.com/mrJeong
.. _Chiheon Kim: https://github.com/chiheonk
.. _Sungbin Lim: https://github.com/sungbinlim
.. _Ildoo Kim: https://github.com/ildoonet
.. _Woonhyuk Baek: https://github.com/wbaek
.. _Boogeon Yoon: https://github.com/bgyoon
.. _Kakao Brain: https://kakaobrain.com/

If you apply this library to any project and research, please cite our code:

.. sourcecode:: bibtex

   @article{kim2020torchgpipe,
       title={torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models},
       author={Chiheon Kim and Heungsub Lee and Myungryong Jeong and Woonhyuk Baek and Boogeon Yoon and Ildoo Kim and Sungbin Lim and Sungwoong Kim},
       year={2020},
       eprint={2004.09910},
       archivePrefix={arXiv}
   }
