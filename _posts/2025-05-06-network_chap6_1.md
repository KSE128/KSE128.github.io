---
published: true
layout: single
title:  "Network : Statistical Models for Network Graphs (1)"
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Statistical Models for Network Graphs' 의 Part 1**에 대한 내용을 담고 있습니다.

---

### Chapter 6 Statistical Models for Network Graphs (Part 1)



#### Introduction

##### Background : Exponential Family Models

확률분포를 다음과 같은 형태로 쓸 경우, Exponential Family에 속한다고 말합니다.



$$
P_\theta(Z = z) = \exp\{\theta^T g(z) - \psi(\theta)\}
$$

$$
\theta : \text{parameter vector} \\
g(z) : \text{a vector of sufficient statistics} \\
\psi(\theta) : \text{log-partition function}
$$



모수 벡터의 경우 분포의 성질을 결정하는 값들을 의미합니다. 충분통계량 벡터의 경우 데이터 z에서 중요한 정보만 요약한 함수를 나타냅니다. 또, 로그-정규화 상수의 경우 확률 분포가 전체에서 1이 되도록 맞춰주는 역할을 합니다. 



$$
\theta^T g(z) : \text{inner product term, normalize}
$$



위의 부분은 데이터와 모수의 상호작용을 나타냅니다. 어떤 데이터가 모수에서 얼마나 영향을 받는지 결정하는 부분이기도 합니다. g(z)는 데이터에서 중요한 정보를 요약한 벡터이고, 모수는 요약 통계량에 각각 얼마나 '무게'를 줄지 결정하는 부분입니다.



위의 확률분포를 요약해서 나타내면, 데이터 z가 관찰될 확률을, 데이터 요약치 g(z)와 모수의 곱으로 계산하고, 정규화 상수로 전체를 확률로 맞춘다고 할 수 있습니다.



이 확률 분포는 베르누이, 가우시안, 푸아송, 정규분포와 같은 유명한 분포들을 하나로 통합해서 설명할 수 있게 됩니다. 

interpretability : 해석 용이성 → 각 모수가 무슨 의미인지 명확하게 해석 가능해지게 됩니다. 

modularity : 모듈성 → 여러 분포나 모형을 조립하듯 확장이 가능해집니다.

connections to statistical inference : 통계 추론과의 연결성 → MLE, 베이지안 추론 같은 통계 기법들이 잘 적용됩니다.



이는 앞으로 다룰 ERGMs (Exponential Random Graph Models)의 배경이 되는 부분입니다. 위에서 제시한 Exponential Family 형태를 네트워크 그래프 전체로 확장한 것입니다. 즉, 그래프의 연결 구조나 네트워크 속성을 g(z)로 요약하고, 그 위에서 확률 모델을 만들어서 네트워크 생성 메커니즘을 이해하거나, 네트워크 데이터를 분석할 수 있게 합니다.



#### Exponential Random Graph Models(ERGMs)

General Formulation



$$
G=(V,E) \text{ with vertex set V and edge set E} \\
Y_{ij}=Y_{ji} : \text{a binary random variable with presence(1), absence(0)}
$$



V는 vertex 집합, E는 edge 집합이 됩니다. Y 부분의 경우 vertex i와 j 사이에 edge이 있으면 1, 없으면 0인 이진 랜덤 변수가 되고, 이를 모아 만든 것이 adjacency matrix Y가 됩니다. y는 Y의 특정한 실현값으로, 관찰된 네트워크의 모습이라고 할 수 있습니다.



$$
P_\theta(Y = y) = \frac{1}{\kappa(\theta)} \exp\left\{ \sum_H \theta_H g_H(y) \right\}
$$

$$
\kappa(\theta) : \text{normalization constant(or partition function)} \\
\kappa(\theta) = \sum_{y \in Y} \exp\left\{ \sum_H \theta_H g_H(y) \right\} \\

\text{the sum runs over all possible graphs y on the given vertex set.}
$$



**Configuration H**은 네트워크 내의 작은 subgraph 패턴 (ex. edge, triangle, star)으로, 전체 그래프 내에서 주목할만한 local structure입니다. 각 configuration H는 우리가 측정하거나 모델링하려고 하는 특정 유형의 구조적 특징에 대응합니다.



네트워크 y 안에서 subgraph H가 존재하는지를 보는 함수로 정의합니다. 만약 존재할 경우 1, 존재하지 않을 경우 0으로 입력합니다.



**Configuration Statistic**


$$
\text{configuration statistic : }g_H(y) \\
g_H(y) = \prod_{(i,j) \in H} y_{ij}
$$


$$
g_H(y) =
\begin{cases}
1 & \text{if subgraph H is present in } y \\
0 & \text{otherwise}
\end{cases}
$$



예를 들어, triangle configuration을 측정할 때,


$$
g_H(y)=y_{ij}y_{jk}y_{ki}
$$


를 이용하여 모두 연결되어 있을 경우, 1 하나라도 연결되어있지 않은 경우 0으로 측정합니다. 



$$
\theta_H : \text{the parameter associated with configuration}\ H
$$



이 모수는 configuration H가 전체 네트워크 구조에 미치는 영향을 나타냅니다. 

만약, 모수가 0이 아니라면, H에 속한 엣지들이 서로 의존성을 가집니다. 엣지들이 의존적이면, 한 edge가 있거나 없는 것이 다른 edge의 존재에 영향을 줍니다. 

H 내에 특정 패턴이 존재할 때
