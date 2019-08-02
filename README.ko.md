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
- PyTorch 1.0 이상

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

### ResNet-101 속도 벤치마크

실험 | torchgpipe | GPipe (논문)
---------- | -----: | -----:
naive-1    |     1x |     1x
pipeline-1 | 0.736x |   0.8x
pipeline-2 | 1.350x | 1.418x
pipeline-4 | 2.291x | 2.182x
pipeline-8 | 3.114x | 2.891x

GPipe 논문의 그림3 (b)에 보고된 ResNet-101 학습 속도 벤치마크를
재현했습니다.

GPipe 없이 한 장치에서 ResNet-101을 학습 시켰을 때 속도인 naive-1을 기준으로
설정했습니다. pipeline-1은 파티션 1개짜리, pipeline-8은 파티션 8개짜리 GPipe로
학습시켰을 때 naive-1 대비 상대속도를 나타냅니다. pipeline-1의 경우 Pipeline
Parallelism이 적용되지 않고 Checkpointing 오버헤드만 있어서 naive-1에 비해
오히려 더 느립니다.

[examples/resnet101_performance_benchmark](examples/resnet101_performance_benchmark)에서
실험 코드를 확인할 수 있습니다.

### AmoebaNet-D 속도 벤치마크

실험 | torchgpipe | GPipe (논문)
---------- | -----: | -----:
naive-2    |     1x |     1x
pipeline-2 | 1.434x | 1.156x
pipeline-4 | 2.049x | 2.483x
pipeline-8 | 2.424x | 3.442x

GPipe 논문의 그림3 (a)에 보고된 AmoebaNet-D 학습 속도 벤치마크 비교에선
torchgpipe와 GPipe간 다소 차이가 있습니다. 이는 TensorFlow로 구현된
AmoebaNet-D를 PyTorch 마이그레이션 과정에서 발생하는 것으로, torchgpipe에 의해
발생한 차이는 아니라고 판단됩니다. 안정된 AmoebaNet-D 재현이 가능할 때 결과를
업데이트 할 예정입니다.

GPipe 없이 두 장치에서 AmoebaNet-D을 학습 시켰을 때 속도인 naive-2을 기준으로
설정했습니다. pipeline-2에서는 논문보다 조금 더 빨랐지만 pipeline-4, 8에서는
느렸습니다.

### AmoebaNet-D 메모리 벤치마크

<table>
  <thead>
    <tr>
      <th rowspan="2">실험</th>
      <th colspan="2">naive-1</th>
      <th colspan="2">pipeline-1</th>
      <th colspan="2">pipeline-2</th>
      <th colspan="2">pipeline-4</th>
      <th colspan="2">pipeline-8</th>
    </tr>
    <tr align="center">
      <td>torchgpipe</td>
      <td>논문</td>
      <td>torchgpipe</td>
      <td>논문</td>
      <td>torchgpipe</td>
      <td>논문</td>
      <td>torchgpipe</td>
      <td>논문</td>
      <td>torchgpipe</td>
      <td>논문</td>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>AmoebaNet-D (L, F)</td>
      <td colspan="2">(6, 208)</td>
      <td colspan="2">(6, 416)</td>
      <td colspan="2">(6, 544)</td>
      <td colspan="2">(12, 544)</td>
      <td colspan="2">(24, 512)</td>
    </tr>
    <tr align="center">
      <td># of Model Parameters</td>
      <td>90M</td>
      <td>82M</td>
      <td>358M</td>
      <td>318M</td>
      <td>613M</td>
      <td>542M</td>
      <td>1.16B</td>
      <td>1.05B</td>
      <td>2.01B</td>
      <td>1.80B</td>
    </tr>
    <tr align="center">
      <td>Total Peak Model Parameter Memory</td>
      <td>1.00GB</td>
      <td>1.05GB</td>
      <td>4.01GB</td>
      <td>3.80GB</td>
      <td>6.45GB</td>
      <td>6.45GB</td>
      <td>13.00GB</td>
      <td>12.53GB</td>
      <td>22.42GB</td>
      <td>24.62GB</td>
    </tr>
    <tr align="center">
      <td>Total Peak Activation Memory</td>
      <td>-</td>
      <td>6.26GB</td>
      <td>6.64GB</td>
      <td>3.46GB</td>
      <td>11.31GB</td>
      <td>8.11GB</td>
      <td>18.72GB</td>
      <td>15.21GB</td>
      <td>35.78GB</td>
      <td>26.24GB</td>
    </tr>
  </tbody>
</table>

GPipe 논문의 테이블1에 보고된 AmoebaNet-D 메모리 효율 벤치마크를 재현했습니다.
AmoebaNet-D 모델은 레이어 수에 비례하는 파라미터 L과 필터 개수에 비례하는
파라미터 F로 모델크기를 조절할 수 있습니다.

한 개의 GPU에서 GPipe를 사용하지 않은 naive-1보다 GPipe를 사용한 pipeline-1에서
더 큰 모델을 학습시킬 수 있는걸 볼 수 있습니다. GPU 개수를 늘린 pipeline-8에선
naive-1 대비 22배 이상 큰 모델도 학습시킬 수 있었습니다.

## 참고사항

이 프로젝트는 개발진이 의도한대로 동작하나, 아직 인터페이스가 확정되지
않았습니다. v0.1.0 전까지는 공개된 API가 경고 없이 바뀔 수 있습니다.

## 개발진 및 사용권

torchgpipe 프로젝트는 [카카오브레인][]의 [이흥섭][], [정명룡][]이 개발하고
[임성빈][], [김치헌][], [김일두][], [백운혁][]의 도움을 받았습니다. [Apache
License 2.0 사용권](LICENSE)으로 배포됩니다.

[카카오브레인]: https://kakaobrain.com/
[이흥섭]: https://subl.ee/
[정명룡]: https://github.com/mrJeong
[임성빈]: https://github.com/sungbinlim
[김치헌]: https://github.com/chiheonk
[김일두]: https://github.com/ildoonet
[백운혁]: https://github.com/wbaek

## 인용

해당 라이브러리를 연구용으로 사용할 경우, 아래 BibTeX 링크를 인용해야 합니다.

```
@misc{torchgpipe,
  author       = {Kakao Brain},
  title        = {torchgpipe, {A} {GPipe} implementation in {PyTorch}},
  howpublished = {\url{https://github.com/kakaobrain/torchgpipe}},
  year         = {2019}
}
```
