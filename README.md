# 군중 계수(Crowd Counting) 모델의 계산 효율성을 위한 경량 모델링
모델링 대상
- Transformer 기반의 군중 계수 SOTA 모델인 PET(Point-Query Quadtree for Crowd Counting, Localization, and More)모델

데이터셋
- ShanghaiTech A(https://paperswithcode.com/dataset/shanghaitech)

제공사항
- 베이스 라인으로서 학습된 군중 계수 모델과 해당 모델을 학습 시킨 데이터 셋 제공
- 모델 정확도 측정을 위한 도구 제공
- 추론 속도 측정을 위한 도구 제공

수행사항
- 모델을 구성하는 레이어, 합성곱 블록 등을 수정하여 베이스 라인 모델의 성능을 유지하면서 CPU/GPU에서의 추론 속도 개선

접근방법 
- 모델을 학습시키는 방법이 아닌 주어진 모델에 대한 구조 변경만을 허용함


## 기간
24.02.26 ~ 24.03.27

## 프로젝트 목표
Transformer 기반의 군중 계수 SOTA 모델인 PET (Chengxin Liu et al., ICCV 2023)모델을 구성하는 레이어/블록을 재설계하여 모델의 정확도(MAE (Mean Absolute Error))을 최대한 유지하면서도, CPU/GPU에서의 추론 속도를 개선

## 설치 및 실행
```shell
git clone https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07.git
cd level2-3-cv-finalproject-cv-07
mkdir data
# https://paperswithcode.com/dataset/shanghaitech - ShanghaiTech A 데이터 다운로드 후 data 디렉토리에 넣기

pip install -r requirements.txt

# train 시
sh ./train.sh
# transformer 다른 메소드로 변경 시
# ./models/transformer/__init__.py 에 from .prog_win_transformer import build_encoder, build_decoder 해당 부분 변경

# eval 시
sh ./evel.sh

```
