# 군중 계수(Crowd Counting) 모델의 계산 효율성을 위한 경량 모델링

<img width="450" src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/83398511/585ff861-3771-42ee-b03d-0fccad6d2896">

모델링 대상
- Transformer 기반의 군중 계수 SOTA 모델인 PET(Point-Query Quadtree for Crowd Counting, Localization, and More)모델

데이터셋
- [ShanghaiTech A](https://paperswithcode.com/dataset/shanghaitech)
  
  <img width="398" alt="image" src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/83398511/6c42593c-c7c4-4cb6-ab91-f6f44c55e8bc">


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

## 프로젝트 팀 구성 및 역할

  |<a href="https://github.com/kimhankyu">김한규 </a>| <a href="https://github.com/haeun1">민하은 </a> | <a href="https://github.com/HayeonLee88">이하연 </a> | <a href="https://github.com/DorianYellow"> 심유승 </a>| <a href="https://github.com/chyeon01">안채연 </a>| <a href="https://github.com/KANG-dg">강동기 </a>| 
  | :-: | :-: | :-: | :-: | :-: | :-: |
  | <img width="100" src="https://avatars.githubusercontent.com/u/32727723?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/87661039?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/83398511?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/146207162?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/86558738?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/121837927?v=4"> |

- 강동기: backbone의 영향이 낮은 layer 구조 변경, transformer의 encoder, decoder구조 개선 (1x1convolution, batch norm, layer감소)
- 김한규: transformer의 encoder layer 구조 개선(FastViT)
- 민하은: vgg13_bn backbone 교체, transformer의 encoder layer 구조 개선(poolformer)
- 심유승: Encoder 블록 재설계 - 블록개수 최적화, window size 최적화, FFN 조정
- 안채연 : vgg11_bn backbone교체, transformer의 encoder layer 구조 개선 (depthwise)
- 이하연: mobilenet backbone 교체, encoder progressive window size 변경, transformer의 encoder, decoder Parameters sharing, transformer의 encoder layer 구조 개선 (poolformer)



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
# 🛠️Methodology

### Backbone 경량화

<img width="800" alt="Untitled" src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/87661039/c6be21a0-2015-4cb0-b22b-616281284790">


- Backbone 교체: mobilenet_v3, vgg11_bn, vgg13_bn
- Backbone layer 제거: 비중이 낮은 Batchnorm layer를 선별, 제거

### Encoder 경량화

1. PoolFormer
<img width="800" alt="image" src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/83398511/34a2a9b9-3c3d-42c9-b47f-cf9f8505a04b">
    
- PET의 Encoder에 self attention 연산을 pooling으로 대체
- 연산을 효율적으로 계산, token mixer 역할 수행
- cross-channel pooling을 사용해 여러 feature map 간 정보 통합

2. Depthwise
    
<img width="800" alt="스크린샷 2024-03-27 오후 12 30 54" src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/87661039/a25b51e4-2eab-4dc1-8ae7-6542ad8b6d3b">
    
- Poolformer 사용 시 성능 하락 보완하기 위해 depthwise convolution 사용
- 지연 시간 오버헤드 도입하지 않으면서 성능 향상

3. Component 재설계
    
<img width="600" alt="스크린샷 2024-03-27 오후 12 39 12" src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/87661039/ac66602d-e12d-47cc-9ae0-5ff01d389c35">
   
- encoder block 개수 최적화
- window size 최적화
- FFN 조정

### Encoder, Decoder 경량화

<img width="400" alt="스크린샷 2024-03-27 오후 12 33 55" src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/87661039/a98863df-4192-4014-acb6-3946ebb07efb">


- encoder, decoder의 linear layer, layer norm 간소화
- Linear layer를 1x1 convolution으로 대체
- Feed froward network 조정

### **TOP 3**

**Mae 측면**

|  | 실험명 | Best MAE | Inference time |
| --- | --- | --- | --- |
| encoder reduction | encoder layer X 2 + [(32,16),(8,4)] | 약 6.22% 감소(50.49→47.35) | 6.92ms 감소(63.95→57.03) |
| poolformer | Enc win size  1/4 + attnX2-> poolingX2 | 약 0.30% 감소(50.13→49.98) | 5.6ms 감소(63.95→58.35) |
| depthwise | depthwise encoder layer 1개 [(8,4)] | 약 0.02% 증가(52.39 → 52.4) | 8.29ms 감소(65.62 → 57.33) |

**Inference 측면**

|  | 실험명 | Best MAE | Inference time |
| --- | --- | --- | --- |
| encoder reduction | encoder layer X 2 + [(32,16),(8,4)]• ffn 제거 | 약 1.78% 증가 (50.49→51.39) | 9.19ms 감소 (63.95→54.76) |
| poolformer | layer reduction + pooling(X2) [(32,16),(8,4)] | 약 6.98% 증가 (50.13→53.63) | 6.98ms 감소 (63.95→56.97) |
| depthwise | depthX1_attnX1 | 약 2.04% 증가 (52.39 → 53.46) | 9.27ms 감소 (63.95 → 54.68) |

### 최종 모델

![pet_A](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/83398511/e66fccce-fcd4-4708-843a-e4b0da9b6ca7)
![our_model](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-07/assets/83398511/4c3bf7d0-1f02-4a5d-8d8e-59d6886dac32)


1. encoder layer를 4개에서 2개로 감소
2. Encoder window size를 [(32,16),(16,8)] 에서 [(32,16),(8,4)]로 변경
3. 인코더 및 디코더에서 FFN을 제거

|  | 실험명 | Best MAE | Inference time |
| --- | --- | --- | --- |
| encoder reduction | encoder layer X 2 + [(32,16),(8,4)] + ffn 제거 | 약 1.78% 증가 (50.49→51.39) | 9.19ms 감소 (63.95→54.76) |

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


