# torchgpipe <img src="docs/_static/not-pipe.svg" height="20" />

[![PyPI](https://img.shields.io/pypi/v/torchgpipe.svg)](https://pypi.org/project/torchgpipe)
[![Build Status](https://travis-ci.org/kakaobrain/torchgpipe.svg?branch=master)](https://travis-ci.org/kakaobrain/torchgpipe)
[![Coverage Status](https://coveralls.io/repos/github/KakaoBrain/torchgpipe/badge.svg?branch=master)](https://coveralls.io/github/KakaoBrain/torchgpipe?branch=master)
[![Documentation Status](https://readthedocs.org/projects/torchgpipe/badge/?version=latest)](https://torchgpipe.readthedocs.io/en/latest/?badge=latest)
[![English README](https://img.shields.io/badge/readme-english-blue.svg)](README.md)

PyTorch 용 [GPipe](https://arxiv.org/abs/1811.06965) 구현입니다. TPU 대신
CUDA를 활용합니다.

```python
from torchgpipe import GPipe
model = nn.Sequential(a, b, c, d)
model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)
output = model(input)
```

## GPipe란?

GPipe는 Google Brain에서 발표한 학습 기법으로, 메모리를 많이 차지하는 큰 모델을
효율적으로 학습시키는 데 유용합니다. Google이 공개한 논문의 벤치마크에 따르면
기준보다 8배 많은 장치(TPU)로 25배 큰 모델을 학습시킬 수 있고, 기준보다 4배
많은 장치에서 3.5배 빨리 학습시킬 수 있다고 합니다.

[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

Google은 GPipe를 이용해 5.6억개의 패러미터를 가지는 AmoebaNet-B 모델을
학습시켰습니다. 이 모델은 ImageNet에서 top-1 정확도 84.3%, top-5 정확도 97.0%로
SOTA를 기록하고 있습니다. (2019년 5월 기준)

GPipe는 Pipeline Parallelism과 Checkpointing, 두 가지 방법으로 가능한 큰 모델을
학습시킵니다.

<dl>
<dt>Pipeline Parallelism</dt>
<dd>우선 GPipe는 모델을 여러 파티션으로 나눠 각각 서로 다른 장치에 배치해 더
    많은 메모리를 사용할 수 있게 한다. 그리고 여러 파티션이 최대한 병렬적으로
    작동할 수 있도록, 모델에 입력되는 미니배치를 여러 마이크로배치로 나눠서
    모델에 흘려보낸다.</dd>

<dt>Checkpointing</dt>
<dd>각 파티션엔 체크포인트를 만들어 메모리 가용량을 극대화한다. 순전파(forward
    propagation) 때 파티션 경계의 입출력만 기억하고 내부의 히든레이어는
    휘발시킨다. 휘발된 히든레이어는 역전파(backpropagation) 때 다시
    계산된다.</dd>
</dl>

## 사용법

현재 torchgpipe는 다음 환경을 지원합니다:

- Python 3.6 이상
- PyTorch 1.1 이상

우선 `torchgpipe`를 PyPI에서 설치합니다:

```sh
$ pip install torchgpipe
```

임의의 `nn.Sequential` 모듈을 `torchgpipe.GPipe`로 감싸면 GPipe가 적용됩니다.
`balance` 인자는 각 파티션의 레이어 개수를 정합니다. `chunks` 인자는
마이크로배치 개수를 설정합니다. 모듈의 입출력과 각 파티션 경계의 입출력은 모두
`Tensor` 혹은 `Tuple[Tensor, ...]` 형식이어야 합니다.

다음 예제코드는 총 4층으로 이뤄진 모듈을 각각 1층씩 지니는 4개의 파티션으로
나누는 방법을 보여줍니다. 마이크로배치 개수는 8개로 설정했습니다:

```python
from torchgpipe import GPipe

model = nn.Sequential(a, b, c, d)
model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)

for input in data_loader:
    output = model(input)
```

## 문서화

API 문서를 비롯한 자세한 문서는 [torchgpipe.readthedocs.io][rtd]에서 확인할 수
있습니다.

[rtd]: https://torchgpipe.readthedocs.io/

## 벤치마크

각 벤치마크의 자세한 내용과 추가적인 벤치마크는
[torchgpipe.readthedocs.io][rtd-benchmarks]에서 확인할 수 있습니다.

[rtd-benchmarks]: https://torchgpipe.readthedocs.io/en/stable/benchmarks.html

### ResNet-101 정확도 벤치마크

배치크기 | torchgpipe | nn.DataParallel | Goyal et al.
-------- | ---------: | --------------: | -----------:
256      | 21.99±0.13 |      22.02±0.11 |   22.08±0.06
1K       | 22.24±0.19 |      22.04±0.24 |          N/A
4K       | 22.13±0.09 |             N/A |          N/A

GPipe를 사용해 학습할 때 추가적인 하이퍼파라미터 조정이 없길 바랍니다. 이를
검증하기 위해 [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)
논문의 표2(c)에 보고된 ResNet-101 정확도(오답률) 벤치마크를 재현했습니다.

### U-Net (B, C) 메모리 벤치마크

실험       | U-Net (B, C) | 파라미터 | 메모리 사용량
---------- | ------------ | -------: | ------------:
baseline   | (6, 72)      |   362.2M |      20.3 GiB
pipeline-1 | (11, 128)    |    2.21B |      20.5 GiB
pipeline-2 | (24, 128)    |    4.99B |      43.4 GiB
pipeline-4 | (24, 160)    |    7.80B |      79.1 GiB
pipeline-8 | (48, 160)    |   15.82B |     154.1 GiB

GPipe로 얼마나 큰 U-Net 모델을 학습시킬 수 있는지 측정했습니다. *baseline*은
GPipe를 적용하지 않은 경우를 나타내고, *pipeline-1*, *-2*, *-4*, *-8*은 GPipe를
적용했을 때 GPU 수에 따른 경우를 나타냅니다.

이 벤치마크엔 간략화한 U-Net 구조를 사용했습니다. 모델의 크기는 하이퍼파라미터
B와 C로 결정합니다. 각각 레이어 수와 필터 수에 비례합니다.

### U-Net (5, 64) 속도 벤치마크

실험       | 처리량   | 속도향상
---------- | -------: | -------:
baseline   | 28.500/s |       1×
pipeline-1 | 24.456/s |   0.858×
pipeline-2 | 35.502/s |   1.246×
pipeline-4 | 67.042/s |   2.352×
pipeline-8 | 88.497/s |   3.105×

U-Net 구조는 여러 롱스킵커넥션을 포함합니다. 스킵커넥션에서의 효율성을 검증하기
위해 U-Net에서 GPU 수에 따른 처리량을 측정했습니다.

### AmoebaNet-D (18, 256) Speed Benchmark

실험      | 처리량    | torchgpipe | Huang et al.
--------- | --------: | ---------: | -----------:
n=2, m=1  |  26.733/s |         1× |           1×
n=2, m=4  |  41.133/s |     1.546× |        1.07×
n=2, m=32 |  47.386/s |     1.780× |        1.21×
n=4, m=1  |  26.827/s |     1.006× |        1.13×
n=4, m=4  |  44.543/s |     1.680× |        1.26×
n=4, m=32 |  72.412/s |     2.711× |        1.84×
n=8, m=1  |  24.918/s |     0.932× |        1.38×
n=8, m=4  |  70.065/s |     2.625× |        1.72×
n=8, m=32 | 132.413/s |     4.966× |        3.48×

(*n*: 파티션 수, *m*: 마이크로배치 수)

[GPipe](https://arxiv.org/abs/1811.06965) 논문의 표2에 보고된 AmoebaNet-D (18,
256) 학습 속도 벤치마크를 재현했습니다. 논문의 *K*를 *n*으로 바꿔 표기했습니다.

## 참고사항

이 프로젝트는 개발진이 의도한대로 동작하나, 아직 인터페이스가 확정되지
않았습니다. v0.1.0 전까지는 공개된 API가 경고 없이 바뀔 수 있습니다.

## 개발진 및 사용권

torchgpipe 프로젝트는 [카카오브레인][]의 [이흥섭][], [정명룡][], [김치헌][]이
개발하고 [임성빈][], [김일두][], [백운혁][], [윤부근][]의 도움을 받았습니다.
[Apache License 2.0 사용권](LICENSE)으로 배포됩니다.

[카카오브레인]: https://kakaobrain.com/
[이흥섭]: https://subl.ee/
[정명룡]: https://github.com/mrJeong
[김치헌]: https://github.com/chiheonk
[임성빈]: https://github.com/sungbinlim
[김일두]: https://github.com/ildoonet
[백운혁]: https://github.com/wbaek
[윤부근]: https://github.com/bgyoon

## 인용

해당 라이브러리를 연구용으로 사용할 경우, 아래 BibTeX 링크를 인용해야 합니다.

```
@article{kim2020torchgpipe,
    title={torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models},
    author={Chiheon Kim and Heungsub Lee and Myungryong Jeong and Woonhyuk Baek and Boogeon Yoon and Ildoo Kim and Sungbin Lim and Sungwoong Kim},
    year={2020},
    eprint={2004.09910},
    archivePrefix={arXiv}
}
```
