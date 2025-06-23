---
published: true
layout: single
title:  "Network : Network Topology Inference (1)"
categories: Network
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Network Topology Inference' 의 Part 1**에 대한 내용을 담고 있습니다.

---

### Chapter 7  Network Topology Inference (Part 1)



#### Introduction

네트워크 그래프는 여러 노드(사람, 컴퓨터, 유전자 등)가 서로 연결된(혹은 연결 안 된) 구조를 나타내는 그림입니다. 이 네트워크의 모양(topology)을 완벽히 알 수도 있고, 일부만 알 수도 있습니다. 우리는 어떤 네트워크에 대해 몇 가지 측정값을 가지고 있을 수 있습니다. x(노드 속성)은 각 노드가 가지고 있는 특징, y(엣지 정보)는 노드끼리 연결되어 있는지(0/1), 둘 다의 정보를 가질 수 있습니다. 

**Network Topology Inference**란, 가능한 여러 개의 그래프(네트워크의 모양) 중에서 "현재 주어진 데이터와 사전 지식을 가장 잘 설명하는 네트워크 구조"가 무엇인지 추론하는 것을 목표로 합니다. 즉, 노드의 속성이나 일부 엣지 정보를 가지고 진짜 네트워크 구조를 찾아내는 과정을 Network Topology Inference이라고 합니다.



우리는 총 3가지의 추론 방법에 대해 다룹니다. 

**Link prediction**은 네트워크의 일부 엣지만 관측된 상황에서 "아직 관측하지 못한 노드쌍이 연결되어 있는지"를 예측하는 과정입니다. 예를 들어, 페이스북에서 몇몇 친구 관계만 알고 있을 때, 나머지 사람들도 친구일지 예측하게 됩니다. **Association Network Inference**은 노드의 속성이 있을 때, 두 노드가 연관성이 충분히 높다고 판단되는 경우에만 엣지가 있다고 간주합니다. 실제 연결이 보이는게 아니라 속성 정보에서 연결을 "추론"하게 됩니다. 예를 들어, 유전자 데이터에서 각각의 유전자 발현 패턴이 비슷한 정도를 계산하여 서로 강하게 연관된 유전자끼리만 연결하는 네트워크를 만듭니다. 

**Tomographic Network Inference**는 네트워크의 경계에 있는 노드들만 측정할 수 있고 내부 구조는 관측할 수 없는 경우에 다룹니다. 이때는 경계에서 얻은 정보만으로 숨겨진 전체 네트워크 구조를 추론해야 합니다. 예를 들어, 인터넷의 몇몇 컴퓨터에서만 데이터를 보내고 받아본 후, 그 경로의 지연 시간 등만 가지고 전체 네트워크의 구조를 알아내는 과정입니다. 

이 세 가지 추론 과정은 얼마나 정보를 직접적으로 알고 있으냐에 따라 다르게 됩니다. 각각의 추론 과정은 점점 정보가 제한되는 상황일 때 다루게 됩니다.

![inference]({{site.url}}\images\2025-06-07-network_chap7_1\inference.PNG)

이 그림은 Network Topology Inference의 세 가지 종류를 시각화한 그림입니다.

먼저, 굵은 실선은 실제로 존재하는 엣지를 말합니다. 점선은 실제로 연결이 없는 경우를 의미합니다. 진한 색은 실제로 관측한 노드와 엣지를 의미하고, 연한 색은 우리가 관측하지 못한(숨겨진) 노드와 엣지를 말합니다.

왼쪽 위는 실제 네트워크 구조를 나타낸 그래프입니다. 오른쪽 위는 Link Prediction 추론으로 나타난 그래프로 일부 노드와 연결만 관측했고, 나머지는 관측하지 못한 경우입니다. 아직 관측하지 못한 노드쌍이 연결되어있는지 예측하는 과정입니다. 왼쪽 아래의 그래프는 Association Network Inference 추론으로 나타난 그래프로 모든 노드의 속성 정보만 있고, 엣지는 관측하지 못했습니다. 연관성이 높은 노드쌍이 연결되어있을지를 추론하는 과정입니다. 오른쪽 아래는 Tomographic Network Inference로 네트워크의 일부 경계 노드만 관측할 수 있고 나머지는 전부 숨겨져있습니다. 이는 관측 가능한 경계(우리가 직접 데이터를 볼 수 있는 노드들)에서만 데이터를 보고 전체 구조를 유추하는 과정입니다.



#### Link Prediction

##### Introduction


$$
Y_{N_v \times N_v}
$$
는 

크기의 이진(0/1) 인접행렬입니다. 여기서 N_v는 네트워크의 노드의 개수이고, Y는 연결될 경우 1, 연결되지 않을 경우 0이 됩니다. 실제로는 Y의 모든 원소를 관측하지 못하고, 일부(Y_obs)만 관측하게 됩니다. 나머지는 Y_miss로 모르는 상태입니다. 우리는 관측된 데이터(Y_obs)와 노드의 특징(X)을 가지고 관측되지 않은 연결이 존재할지를 예측하는 과정입니다. X는 노드의 속성 정보 (나이, 취미, 유전자 정보)를 나타냅니다. 

Missing Data의 이유는 다양합니다. 이는 시간에 따라 바뀔 수도 있고, 샘플링의 방법때문일 수도 있습니다. 그래서 우리는 Link Prediction의 기본 가정으로 Missing at Random(MAR)을 가정합니다. MAR 가정이란, 관측이 안된 데이터가 우리가 이미 관측한 다른 값들에만 의존해서 빠졌을 뿐, 그 값 자체(Y)와는 무관하다는 의미를 담고 있습니다. MAR 가정이 있어야 우리가 관측된 데이터만으로 공정하게 나머지를 예측할 수 있습니다. 만약 MAR이 깨지면, 그 예측은 왜곡될 수 있습니다.



##### Model-Based Prediction

관측된 네트워크 데이터와 노드 속성을 바탕으로 우리가 미처 관측하지 못한 연결이 있을 확률을 모델(확률적/통계적) 을 사용하여 예측하는 방식입니다. 

이때 예측한다는 것은 


$$
P(Y_{miss} \mid Y_{obs}=y_{obs}, \ X=x)
$$


의 확률을 가지는데, 관측된 정보가 주어졌을 때 미관측 연결이 어떤 값을 가질 확률분포를 구한다는 의미를 담고 있습니다. 네트워크 전체의 연결을 한 번에 예측하는 것은 복잡하기 때문에, 실제로는 "각 노드쌍 별"로 예측에 대한 개별적인 계산을 진행하는 경우가 많습니다. 

여기서 Y_miss의 개수는


$$
N(Y_{miss})=\frac{N_v(N_v-1)}{2}
$$


로 miss된 노드들을 하나하나씩 모두 predict하게 됩니다. 결국, 계산 횟수가 굉장히 많아지게 됩니다.



이 방식은 교차검증(cross-validation)에도 쓰이는데, 데이터를 일부만 감추고 나머지 데이터로 학습을 시킨다음, 감춘 부분을 예측해 모델 성능을 평가합니다. 이 과정에서 fold assignment(데이터를 나누는 방법)는 연결의 유무와 무관하게 무작위이기 때문에 misssing at random (MAR) 가정에 부합하게 됩니다. 각각의 빠진 Y에 대해 모델로부터 예측 확률값(posterior expectation)을 구합니다. 이 프레임워크는 link prediction 문제와 자연스럽게 맞고, 모델의 성능을 공정하게 평가할 수 있도록 도웁니다.

만약, 값이 크면 missing한 Y에 대한 값이 0, 그 반대의 경우 1이 됩니다.



##### Scoring Methods

관측되지 않은 노드쌍(i,j)마다 둘이 연결될 가능성을 점수로 계산하는 방식으로 각 쌍마다 점수 s(i,j)를 계산하여 점수가 일정 기준(threshold) 이상이거나, 점수가 높은 쌍부터 순위별로 골라서 둘의 연결 예측 가능성을 선택하게 됩니다.



1. Shortest-path Score (최단경로 점수) 

   두 노드 사이의 최단 거리가 짧을수록 새로운 엣지가 생길 확률이 높다고 가정합니다. 거리가 가까울수록 (점수 절댓값이 작을수록) 연결될 가능성이 높습니다.

   
   $$
   s(i,j)=-dist_{G_{Obs}}(i,j)
   $$
   
   
   
2. **Common neighbors (공통 이웃 수)**

   i와 j가 공통으로 연결된 이웃의 수가 많을수록 둘의 연결될 가능성이 높다고 봅니다.

   여기서 N_i는 i의 관측된 이웃 집합을 의미합니다.

   
   $$
   s(i, j) = \left| N_i^{\mathrm{obs}} \cap N_j^{\mathrm{obs}} \right|
   $$
   



3. Jaccard coefficient (자카드 계수)

   공통 이웃의 비율을 보는 지표입니다. 즉, 이웃이 겹치는 정도(유사도)를 0~1 사이의 값으로 정규화해서 사용합니다.

   
   $$
   s(i, j) = \frac{ \left| N_i^{\mathrm{obs}} \cup N_j^{\mathrm{obs}} \right| }{ \left| N_i^{\mathrm{obs}} \cap N_j^{\mathrm{obs}} \right| }
   $$

4. 





​	Common neighbors와 Jaccard coefficient는 비슷한 성격을 가집니다. 

​	또, Jaccard coefficient는 가장 일반적인 score 계산 방법입니다.





4. Adamic-Adar score (아다믹-아다 점수)

   공통 이웃 중 드물게 등장하는(희귀한) 이웃일수록 더 큰 가중치를 부여합니다.

   
   $$
   s(i, j) = \sum_{k \in N_i^{\mathrm{obs}} \cap N_j^{\mathrm{obs}}} \frac{1}{\log |N_k^{\mathrm{obs}}|}
   $$
   

​	이웃 k가 연결이 적을수록, 점수가 더 커지는 시스템입니다.



다음은 fblog에서 Common Neighbors을 구하는 R코드입니다.

```R
library(sand)
nv <- vcount(fblog)
ncn <- numeric()
A <- as_adjacency_matrix(fblog)
for(i in 1:(nv-1)){
  ni <- neighborhood(fblog, 1, i)
  nj <- neighborhood(fblog, 1, (i+1):nv)
  nbhd.ij <- mapply(intersect, ni, nj, SIMPLIFY=FALSE)
  temp <- unlist(lapply(nbhd.ij, length)) - 2 * A[i, (i+1):nv]
  ncn <- c(ncn, temp)
}

# 실제 엣지의 존재 유무에 따라 분리
Avec <- A[lower.tri(A)]

# 분포 시각화
library(vioplot)
vioplot(ncn[Avec==0], ncn[Avec==1], 
        names=c("No Edge", "Edge"))
title(ylab="Number of Common Neighbors")
```

<img src="{{site.url}}\images\2025-06-07-network_chap7_1\common.PNG" alt="common" style="zoom:67%;" />

이 plot에서 각 그룹의 폭이 넓은 곳은 공통 이웃 수를 가지는 노드쌍이 많다는 것을 의미합니다. 엣지가 없는 쌍에서는 공통 이웃 수가 적은 쌍이 많고, 분포가 아래쪽(0-5)에 집중되어 있습니다. 그러나, 엣지가 있는 쌍의 경우 공통 이웃 수가 더 크고, 전체적으로 위쪽으로 퍼져있는 모습을 볼 수 있습니다. 10개, 20개, 많게는 30개 넘게 공통 이웃이 있는 경우도 많습니다. 이 플롯은 실제로 연결된 노드쌍은 공통 이웃이 많은 경우가 훨씬 많습니다. 연결이 없는 쌍은 대부분 공통 이웃이 적은 것을 알 수 있습니다. 즉, **공통 이웃의 수가 많으면 실제로 엣지가 있을 확률이 큽**니다. 이 특성이 link prediction에 활용될 수 있습니다.



```R
> library(ROCR)
> pred <- prediction(ncn, Avec)
> perf <- performance(pred, "auc")
> slot(perf, "y.values")
[[1]]
[1] 0.9275179
```



위의 R코드는 Common Neighbors Score로 link prediction을 했을 때, 모델의 성능을 평가한 것으로 AUC값을 나타냅니다. AUC(ROC curve 아래의 면적)은 모델의 구분력을 나타내는 지표로, 0.9275179라는 값은 매우 높은 수치임을 알 수 있습니다. 즉, 공통 이웃 수만으로도 실제 네트워크에서 엣지가 있을지 아닐지를 정확하게 예측할 수 있습니다.

