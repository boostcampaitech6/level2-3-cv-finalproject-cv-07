# 군중 계수(Crowd Counting) 모델의 계산 효율성을 위한 경량 모델링
모델링 대상
- Transformer 기반의 군중 계수 SOTA 모델인 PET(Point-Query Quadtree for Crowd Counting, Localization, and More)모델

데이터셋
- [ShanghaiTech A](https://paperswithcode.com/dataset/shanghaitech)

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
Transformer 기반의 군중 계수 SOTA 모델인 PET (Chengxin Liu et al., ICCV 2023)모델을 구성하는 레이어/블록을 재설계하여 모델의 정확도(MAE(Mean Absolute Error))을 최대한 유지하면서도, CPU/GPU에서의 추론 속도를 개선

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
### 최종 모델

![pet_A](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/83398511/e66fccce-fcd4-4708-843a-e4b0da9b6ca7)
![our_model](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/83398511/4c3bf7d0-1f02-4a5d-8d8e-59d6886dac32)


1. encoder layer를 4개에서 2개로 감소
2. Encoder window size를 [(32,16),(16,8)] 에서 [(32,16),(8,4)]로 변경
3. 인코더 및 디코더에서 FFN을 제거

|  | 실험명 | Best MAE | Inference time |
| --- | --- | --- | --- |
| encoder reduction | encoder layer X 2 + [(32,16),(8,4)] 
+ ffn 제거 | 약 1.78% 증가 
(50.49→51.39) | 9.19ms 감소 
(63.95→54.76) |

mae 측면에서 **성능하락이 1.78% 수준이며 inference time은 9.19ms** 대폭 감소하였기에

mae와 inference time 측면 모두에서 top 3의 수준을 기록하였다.

**mae : 51.39 inference time 54.76ms로 최종 모델로 선정 되었다.**

## References
1) [Liu, Chengxin, et al. "Point-query quadtree for crowd counting, localization, and more."](https://arxiv.org/pdf/2308.13814.pdf)
2) [Yanyu Li, Ju Hu, et al. "Rethinking Vision Transformers for MobileNet Size and Speed."](https://arxiv.org/pdf/2212.08059.pdf)
3) [Howard, Andrew, et al. "Searching for mobilenetv3."](https://arxiv.org/pdf/1905.02244.pdf)
4) [Liu, Chengxin, et al. "Point-query quadtree for crowd counting, localization, and more."](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Liu_Point-Query_Quadtree_for_ICCV_2023_supplemental.pdf)
5) [Yu, Weihao, et al. "MetaFormer Is Actually What You Need for Vision"](https://arxiv.org/pdf/2308.13814.pdf)
6) [Li, Yanyu, et al. "Efficientformer: Vision transformers at mobilenet speed."](https://arxiv.org/pdf/2206.01191.pdf)


