---
published: true
layout: single
title:  "Network : Statistical Models for Network Graphs (1)"
categories: Network
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

**General Formulation**



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

즉, 해당 패턴 H이 실제 네트워크 y에 완전히 포함되어 있는지를 판단하는 방법이라고 생각하면 됩니다.

예를 들어, triangle configuration을 측정할 때,

$$
g_H(y)=y_{ij}y_{jk}y_{ki}
$$

를 이용하여 모두 연결되어 있을 경우, 1 하나라도 연결되어있지 않은 경우 0으로 측정합니다. 

삼각형 구성 H : 노드 i, j, k 사이에 세 엣지 (i,j), (j,k), (k,i)가 존재해야합니다.



$$
\theta_H : \text{the parameter associated with configuration}\ H
$$

이 모수는 구성 H (edge, triangle, star)에 연결되어 있습니다. 이 값은 해당 구성이 전체 네트워크 구조에 얼마나 영향을 미치는지를 나타냅니다.

만약 모수가 0이 아니라면, H 안에 포함된 엣지들의 존재 여부가 **서로 영향을 주게 됩니다**. 즉, 한 엣지가 있거나 없다는 사실이 다른 엣지의 존재에 영향을 줄 수 있습니다.

앞의 예시에 이어서 H가 삼각형이라고 할 때, **모수의 크기가 0보다 크다면**, 네트워크는 삼각형을 자주 만들고 싶어하는 경향이 있습니다. 이 내용은 노드 i와 j, j와 k가 연결되어 있다면, i와 k도 연결될 가능성이 높아집니다. 엣지들이 서로 독립적으로 생기는게 아니라, 어떤 엣지가 생기면 **다른 엣지도 생길 획률이 바뀌**게 됩니다. 


$$
P_\theta(Y = y) = \frac{1}{\kappa(\theta)} \exp\left( \theta_1 g_1(y) + \cdots + \theta_p g_p(y) \right)
$$
만약 **모수가 0라면**, H가 네트워크 확률 분포에 **영향을 주지 않습니다**. 이는 H에 포함된 엣지들 사이에 의존성이 없고 해당 configuration H를 모델에 고려하지 않겠다고 판단한 것입니다. 모델은 H의 존재를 "중요하지 않다"고 판단하게 됩니다.



ERGM에서는 각 subgraph 패턴과 연결되어 있는 coffeffcient값에 따라서 모수 값이 결정됩니다. 즉, 이 모수는 해당 패턴이 네트워크에서 얼마나 자주 등장하려는 경향이 있는지를 나타냅니다. 모델이 네트워크 안에서 그 구성 H를 얼마나 선호하거나 회피하는지를 나타나게 되는 것입니다.



**Interpretation and Dependence Structure**

먼저, ERGM은 복잡한 의존 구조를 표현할 수 있습니다. 

일반적인 간단한 모델 (예: Erdős–Rényi 모델)은 엣지들 간에 독립적으으로 존재한다고 가정하지만, ERGM은 엣지 하나의 존재가 다른 엣지의 존재 여부에 영향을 줄 수 있는 모델입니다. 엣지들 간의 의존성은 모수와 그에 해당하는 구성 H을 통해 포착됩니다. 
$$
\text{If configuration } H \text{ is a triangle and } \theta_H > 0,  \\ \text{ then the network tends to favor forming triangle structures,} \\ \text{ i.e., transitive closure becomes more likely.}
$$
만약 구성 H가 삼각형이고, 모수값이 0보다 클 때, 두 엣지가 있을 때 세번째 엣지가 생길 확률이 증가해서 삼각형(=transitive closure)을 만들려는 경향이 생깁니다. 



두번째로 ERGM은 조건부 독립성을 가정할 수도 있습니다. 

특정한 제 3의 엣지 집합만 알고 있으면, 다른 두 엣지 집합들이 조건부로 독립이 될 수 있습니다. ERGM은 일반적으로 엣지들 사이에 의존성을 허용하는 모델인데, 일부 엣지들만 조건으로 주어졌을 때 나머지 엣지들끼리는 독립적으로 취급할 수 있습니다.

만약, C라는 세번째 엣지 집합을 이미 알고 있는 조건에서 
$$
C=\{Y_{23}, Y_{24}\}
$$
: 네트워크의 중간에 있는 연결 정보



엣지들의 모음 A , 예를 들어 "노드 1-2, 1-3" 같은 엣지들을 포함한 집합들은
$$
A=\{Y_{12}, Y_{13}\}
$$
 : 노드 1과 관련된 엣지들



B집합의 엣지들과 A집합의 엣지들이 서로 독립일 수 있습니다. 
$$
B = \{Y_{45}, Y_{56}\}
$$
: 완전히 다른 영역의 엣지들



조건부 독립성은 네트워크의 **구성(configuration)** 선택에 따라 달라집니다.

(즉, 어떤 패턴들을 모델에 포함시키느냐에 따라 A와 B가 조건부 독립인지 아닌지가 바뀔 수 있습니다.)



**Extension to Attributes and Directed Graphs**

보통 ERGM은 속성 정보나 방향성이 있는 그래프를 다루지는 않지만, 이 요소들이 있는 그래프도 ERGM으로 모델링할 수 있습니다.

먼저, 방향성이 있는 그래프를 다룰 경우,
$$
Y_{ij} \neq Y_{ji}
$$
둘의 관계는 별개의 관계로 가정합니다. i가 j에 연결됐다고 해서 j가 i에 연결하는 것은 아니게 됩니다.



또, 노드가 가지고 있는 정보들(속성들)을 반영할 수 있습니다.
$$
P_{\theta}(Y = y \mid X = x) \propto \exp(\theta^T g(y, x))
$$
이 식은 속성 정보 X가 주어졌을 때, 어떤 네트워크 Y=y가 나타날 확률을 나타냅니다.



먼저,
$$
P_{\theta}(Y = y \mid X = x)
$$
는 우리가 모델링하고자 하는 대상으로 속성 정보 x가 주어졌을 때, 네트워크 y가 관측될 확률로 조건부 확률입니다.



ERGM의 핵심 부분인 
$$
\exp(\theta^T g(y, x))
$$
에서는 네트워크 구조의 속성 정보를 기반으로 각각의 y가 얼마나 가능성 있는지 결정합니다.


$$
g(y,x)
$$


은 통계량 벡터로 네트워크 y와 속성 x로부터 계산됩니다.

이 항목은 network structure와 attribute similarity 를 포함합니다.

이 수식은 패턴들이 **어떻게 네트워크의 생성 확률에 영향을 주는지**를 수학적으로 표현한 것입니다.



이러한 



XXX가 노드 속성들을 나타내는 벡터(혹은 행렬)라고 하면,
 이 속성을 활용해서 **속성 기반 연결 경향성 (예: 동질성, homophily)**을 모델에 넣을 수 있어요.
