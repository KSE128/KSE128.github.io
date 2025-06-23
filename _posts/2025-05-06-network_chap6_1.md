---
published: true
layout: single
title:  "Network : Exponential Random Graph Models(ERGMs)"
categories: Network
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Exponential Random Graph Models(ERGMs)'**에 대한 내용을 담고 있습니다.

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


C : 네트워크의 중간에 있는 연결 정보



엣지들의 모음 A , 예를 들어 "노드 1-2, 1-3" 같은 엣지들을 포함한 집합들은


$$
A=\{Y_{12}, Y_{13}\}
$$


 A : 노드 1과 관련된 엣지들



B집합의 엣지들과 A집합의 엣지들이 서로 독립일 수 있습니다. 


$$
B = \{Y_{45}, Y_{56}\}
$$


B : 완전히 다른 영역의 엣지들



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



이렇게 유연하게 이뤄지는 ERGM은 모델링하는데에 강한 도구로 만들어지게 됩니다.



##### ERGM's simple Example

 **Erdős–Rényi 모델**

ERGM의 가장 단순한 형태는 **Bernoulli 그래프** 모델입니다. 이 모델은 **Erdős–Rényi**로 알려져 있습니다. 

그래프에 있는 모든 엣지들은 서로 전혀 영향을 주지 않고 독립적으로 생기거나 사라집니다. 또, 각 엣지의 존재 여부만이 중요하고 엣지들 사이의 구조적인 상호작용은 모델에 포함하지 않게 됩니다.


$$
P_{\theta}(Y = y) = \frac{1}{\kappa} \exp\left( \sum_{i, j} \theta_{ij} y_{ij} \right)
$$


이 수식은 엣지만 고려하고, 삼각형, 별 모양 등 3개 이상의 노드를 포함하는 모든 구성 패턴 H에 대해 모수를 0으로 설정한다는 의미입니다. 즉, 네트워크의 구조적 특성은 무시하고, 엣지의 존재 여부만 독립적으로 고려하는 모델입니다.



엣지의 존재 여부만 독립적으로 고려하기 때문에 이 수식을 아래와 같은 형태로 바꿀 수 있습니다.


$$
P_{\theta}(Y = y) = \frac{1}{\kappa} \prod_{i,j} \exp\left( \sum_{i, j} \theta_{ij} y_{ij} \right)
$$




**Erdős–Rényi's Configuration H**

Erdős–Rényi 모델은 각 노드 쌍
$$
(i,j)
$$
에 대해 구성 
$$
H=\{(i,j)\}
$$
인 단 한 개의 엣지, 즉 노드 i와 j 사이의 연결 하나입니다. 모든 구성 H는 단순히 하나의 엣지로 구성된 패턴입니다.

이 configuration statistic은 
$$
g_H(y)=y_{ij}
$$
로 노드 i와 j 사이에 엣지가 실제로 존재하면 1, 존재하지 않으면 0이 됩니다.

그래서 
$$
g_H(y)=y_{ij}
$$
라는건 해당 엣지가 있는지 없는지를 나타내는 값이 나타내는 값이 됩니다.

이는 각 엣지가 존재하냐 아니냐만이 모델의 전부가 되는 것입니다.





이 모델은 각 노드 쌍 
$$
(i,j)
$$
에 대해 구성
$$
H=\{(i,j)\}
$$
를 포함하고 있습니다. 즉, 모든 노드 쌍에 대해 하나의 엣지를 구성으로 취급하고 더 이상 다른 구성은 포함하지 않습니다. 각 H는 단 하나의 엣지만 포함하므로 서로 독립적입니다. → disjoint

엣지들 사이의 상호작용이 없습니다. → single edge만 포함


$$
\text{The model includes such a configuration} \\ \text{for every pair }
1 \leq i < j \leq n
$$


모든 노드 쌍 (i,j)에 대해 해당 엣지를 하나의 구성으로 보고 이 구성들이 전부 ERGM에 포함됩니다.

이러한 이유로 위에 제시한 단순한 형태로 나타날 수 있습니다.


$$
P_{\theta}(Y = y) = \frac{1}{\kappa} \exp\left( \sum_{i, j} \theta_{ij} y_{ij} \right)
$$


만약, 네트워크 전체에서 동질성을 가정한다면, 모든
$$
i<j
$$
에서 
$$
\theta_{ij}=\theta
$$
가 됩니다.


$$
P_{\theta}(Y = y) = \frac{1}{\kappa} \exp\{\theta L(y)\} \ , \\
\text{where } \ L(y) = \sum_{i, j} y_{ij}=N_v
$$
모든 모수가 동일한 값을 가진다면, 네트워크 y의 확률은 오직 **엣지 수**에만 의존하고 그 외 어떤 구조도 고려하지 않습니다. 이는 고전적 모델인 Erdős–Rényi 모델
$$
G(n,p)
$$
와 동일해집니다.



**Erdős–Rényi 모델에서 θ와 p의 관계**


$$
P(Y=y)=p^{L(y)}(1-p)^{\binom{n_v}{2}-L(y)}
$$
여기서 
$$
L(y)
$$
는 네트워크 y 안에 있는 엣지의 총 개수이고, 
$$
\binom{n_v}{2}
$$
는 전체 가능한 엣지의 수를 의미합니다.



이 수식은 각 엣지가 독립적인 베르누이 분포를 따른다는 걸 보여줍니다.

위의 수식을 exponential family 형태로 바꿔보면 다음과 같은 형태의 수식이 나옵니다.


$$
P(Y=y)= \text{exp} \left\{ L(y) \ \log \left(\frac{p}{1-p}\right)+ \binom{n_v}{2} \log (1-p) \right\}
$$


위의 형태는 우리가 알고 있는 exponential family 형태인 


$$
P_{\theta}(Y = y) \propto \exp(\theta \cdot L(y))
$$


처럼 보이게 됩니다.



그리고, ERGM의 단순형 수식인 


$$
P_{\theta}(Y = y) = \frac{1}{\kappa} \exp\{\theta L(y)\}
$$


과 비교해보면, 


$$
\theta= \log \left(\frac{p}{1-p}\right) \ \rightarrow \ p= \frac{e^\theta}{1+e^\theta}
$$


의 형태로 바꿀 수 있게 됩니다.

즉, **ERGM의 파라미터 θ**는 Erdős–Rényi에서의 엣지 존재 확률 p의 **로그 오즈(log-odds)**입니다.

이 식을 통해 ERGM 파라미터 θ를 확률 p로 쉽게 변환할 수 있습니다. : 로지스틱 함수 형태로 변화하게 됩니다.



##### Higher-Order Structures

지금까지의 모델에서는 엣지의 개수만을 고려했는데, 현실 네트워크에 적용하기에는 한계가 존재합니다. 그래서 엣지 뿐만 아니라 좀 더 복잡한 패턴인 star configurations나 triangles을 모델에 포함시키려고 하는 모델이 존재합니다.

이에 등장한, Higher-Order Structures입니다. ERGM 안에 복잡한 구조를 설명하는 통계량을 추가하려고 하는데, 대표적으로 두 가지를 추가합니다.
$$
S_k(y) : \text{numbers of k-stars} \\
T(y) : \text{number of triangles}
$$


지금부터는 엣지의 개수만 세는게 아니라 엣지들이 어떻게 모여 구조를 이루는가를 모델에 넣을 것입니다. ERGM이 더 현실적인 네트워크 생성 메커니즘을 설명할 수 있게 확장하는 과정입니다.


$$
P_{\theta}(Y = y) = \frac{1}{\kappa} \exp \left\{ \sum_{k=1}^{N_v - 1} \theta_k S_k(y) + \theta_\tau T(y) \right\}
$$


위의 수식은 "엣지 수" 뿐만 아니라 star 모양과 triangle 구조의 빈도까지 반영한 확장된 ERGM입니다.

보통의 경우 4-star 이상은 무시합니다. 이는 계산이 복잡해질 수 있고 과적합(overfitting)의 위험성이 존재하기 때문입니다.

Higher-Order를 적용하는 접근은 Model degeneracy를 일으킬 수 있습니다. 이는 모델이 비현실적인 그래프에 확률을 몰아주는 현상으로 예를 들어 완전히 연결된 그래프 (complete) 또는 아예 비어 있는 그래프 (empty) 상태를 말하는데 이런 경우 실제 네트워크와 전혀 다른 극단적인 결과만 생성하게 됩니다.

그렇다면, 더 높은 차수의 star 패턴을 제대로 반영하는 방법은 무엇이 있을까요?



##### Solution of Model Degeneracy

**Alternating k-Star Statistic (AKS)**

여러 개의 k-star를 하나의 단일 통계량으로 요약하되, 그 영향력을 점점 줄이고, 부호를 번갈아 바꿔서 안정적으로 반영하자는 내용입니다.


$$
\text{AKS}_\lambda(y) = \sum_{k=2}^{N_v - 1} \frac{(-1)^k S_k(y)} {\lambda^{k-2}} , \quad \lambda \geq 1
$$


이 수식은 여러 고차 star 효과를 균형 있게 반영하면서, 높은 차수에 너무 큰 영향력을 주지 않도록 제어해줍니다.

k차수가 올라가면서,
$$
\theta_k
$$
에 해당하는 절댓값은 줄어들게 됩니다.

그 이유를 차근차근 살펴보면, 
$$
\lambda^{k-2}
$$
는 k가 커질수록 점점 커집니다.  λ가 1 이상의 값을 가지기 때문입니다. 

따라서,
$$
\frac{S_k(y)}{\lambda^{k-2}}
$$
의 값은 점점 작아집니다. 즉, 고차 star일수록 그 항의 기여도가 줄어들게 됩니다.



각 star의 계수를 각각 추정하지 않고, 대신 아래의 규칙을 따르도록 미리 모양을 설정한 수식입니다.


$$
\theta_k \propto (-1)^k \lambda^{-(k-2)},\quad \text{for all } k \geq 2
$$


이 수식에서는 계수를 따로 따로 추정하지 않고, 규칙을 부여합니다. 이는 모델이 너무 많은 자유도를 갖지 않도록 제약을 주는 **parametric constraint**라고 합니다. 

또, 
$$
\lambda \geq 1
$$
이므로, 
$$
\lambda^{-(k-2)}
$$
는 k가 커질수록 작아지게 됩니다. 즉, 고차 star의 기여도가 줄어들면서 모델 안정성이 높아지게 됩니다.


$$
(-1)^k
$$
는 작은 star와 큰 star의 영향력을 균형 있게 조절하면서 너무 촘촘하거나 너무 희소한 그래프에 확률이 쏠리지 않도록 도와줍니다. 이는 과적합(overfitting)을 방지합니다.



**Equivalent Alternative: Geometrically Weighted Degree (GWD)**

다른 대안은 노드의 degree 분포에 기반한 통계량입니다. 
$$
\mathrm{GWD}_\gamma(y) = \sum_{d=0}^{N_v - 1} e^{-\gamma d} \cdot N_d(y)
$$

| 항목    | 의미                                                      |
| ------- | --------------------------------------------------------- |
| N_d(y)  | **degree가 d**인 노드의 개수                              |
| γ       | **감쇠 계수(decay rate)** – degree가 클수록 가중치 작아짐 |
| e^{−γd} | degree d에 대한 가중치. **지수적으로 줄어듦**             |



감쇠율을 정의하면, 



$$
\gamma = \log\left( \frac{\lambda - 1}{\lambda} \right),\quad \lambda \geq 1
$$



이 정의는 위의 AKS에서 사용된 λ와 동일하게 조절되도록 설계된 것으로 λ가 클수록 γ는 작아지고 가중치 감소가 느려집니다. 

이는 고차 star들을 degree 분포를 통해 간접적으로 요약해서 계산합니다. degree가 높을수록 해당 노드가 여러 star 구성에 관여할 가능성이 높기 때문에, degree 기반 요약(GWD)도 star 기반 요약(AKS)과 매우 유사한 정보를 담고 있습니다.



Alternating k-Star (AKS)도 결국 degree 분포로 표현할 수 있습니다.

먼저 k-star 개수를 degree로 표현합니다.



$$
S_k(y) = \sum_{d=k}^{N_v - 1} \binom{d}{k} N_d(y)
$$

$$
N_d(y)=\text{the number of nodes of degree} \ d
$$



degree가 d인 노드는 총 (d개 중 k개를 뽑은 경우의 수) 개의 k-star를 만들 수 있고, 그런 노드가 N개 있으니, 전체 k-star 개수를 쓸 수 있게 됩니다. 즉, star 개수는 degree 분포를 기반으로 계산이 가능하게 됩니다.

그렇다면, AKS를 degree 기반으로 다시 표현할 수 있게 됩니다.



$$
\text{AKS}_\lambda(y) = \sum_{d=2}^{N_v-1} N_d(y)\sum_{k=2}^{d} (-1)^k \frac{\binom{d}{k}} {\lambda^{k-2}}
$$



각 degree d를 기준으로, 해당 degree가 만들 수 있는 모든 k-star에 대해 alternating weight를 곱해서 더합니다. 

즉, AKS는 degree 분포를 기반으로 한 가중 다항식 합 (weighted polynomial sum)입니다.

AKS는 여러 차수의 star를 모두 고려하되, 직접적으로 star의 수로 표현하지 않고 더 간접적인 정보인 degree 분포를 통해 표현할 수 있습니다.


$$
\mathrm{GWD}_\gamma(y) = \sum_{d=0}^{N_v - 1} e^{-\gamma d} \cdot N_d(y)
$$


와 비교하면,


$$
w(d;\lambda)=\sum_{k=2}^{d} (-1)^k \frac{\binom{d}{k}} {\lambda^{k-2}}\ \propto \ e^{-\gamma d} \ \ \text{with} \ \ \gamma=\log\left(\frac{\lambda-1}{\lambda}\right)
$$


의 관계를 나타낼 수 있습니다.

이 함수는 degree d에 대해 적용되는 감쇠(weight decay) 함수입니다.  λ가 클수록 γ는 작아지고 감쇠가 느려집니다.  

이 weight 함수 
$$
w(d;\lambda)
$$
는 AKS에서 쓰이는 복잡한 **이항계수 기반 가중합**을 **매우 근사적으로 표현**할 수 있습니다.


$$
\sum_{k=0}^{d} \binom{d}{k}(-1)^kx^k=(1-x)^d
$$


여기서 
$$
x=\frac{1}{\lambda}
$$
와 같은 형태를 넣으면 AKS에서 쓰이는 가중치와 거의 비슷한 형태가 됩니다.



결국엔, AKS에서의 weight 구조가 **지수 함수 기반 감쇠 함수**와 유사하다는 점을 보여주며, **AKS와 GWD의 이론적 유사성**을 수학적으로 정당화합니다.



**Triangle : Alternating k-Triangle Statistic (ATS)**


$$
AKT_\lambda(y)=3T_1+\sum_{k=2}^{N_v-2}\frac{(-1)^{k+1}T_k(y)}{\lambda^{k-1}}, \\
T_k \text{ is the number of triangles (sets of k triangles sharing a base)}
$$
T_k는 동일한 엣지를 공유하는 k개의 삼각형 묶음을 이야기합니다. 즉, 한 엣지 주변에 삼각형이 여러 개 겹치는 구조를 의미합니다.

AKT 통계량을 쓰면, 여러 triangle 구조를 하나의 요약된 지표로 처리하게 됩니다. 감쇠와 alternating sign으로 복잡한 클러스터 구조를 완화시켜 더 안정적이고 현실적인 클러스터링 모델링이 가능하게 됩니다.



**+ GWESP (Geometrically Weighted Edgewise Shared Partners)**

ESP는 두 노드가 엣지로 연결되어 있을 때, 그들이 공유하는 친구의 수로 GWESP는 이런 shared partner 수에 지수적 감쇠(weight)를 적용해서 삼각형 경향성을 포착하는 통계량입니다. 고차 삼각형에 덜 민감하게 하여 모델의 안정성과 해석 가능성을 높이는 방식입니다.



##### Adding Vertex Attributes

ERGM은 네트워크 구조뿐만 아니라, 노드 간의 속성 차이(성별, 부서, 연차 등)가 연결에 영향을 미칠 수 있다는 사실도 함께 모델링할 수 있습니다.


$$
g(y, x) = \sum_{1 \leq i < j \leq N_v} y_{ij} \cdot h(x_i, x_j)
$$


이 통계량은 "속성 정보가 연결에 미치는 총 효과"를 정리한 형태입니다. y_i,j는 노드 i와 j 사이에 있지가 있는지의 여부를 나타냅니다. 이 통계량은 속성 정보가 연결에 미치는 총 효과를 나타냅니다.


$$
h(x_i, x_j)
$$
에는 Main effect와 Second-order effect (Homophily effect), 두 가지 효과가 있습니다. 

먼저 Main Effect는 다음과 같은 형태를 나타냅니다.


$$
h(x_i,x_j)=x_i+x_j
$$


이는 연결된 노드들의 속성 수준의 합을 나타냅니다. 또, 속성 수준 자체가 얼마나 연결에 기여하느냐를 측정하는 도구가 됩니다.



두번째는 Homophily effect로 다음과 같은 형태를 나타냅니다.


$$
h(x_i,x_j)=I\{x_i=x_j\}
$$


이 형태의 경우 두 노드의 속성이 같으면 1, 다르면 0으로 동질성을 모델링합니다.



첫번째 (Main effect)는 "속성 자체가 영향을 주는지"를 모델링하고, 두번째(homophily effect)는 "속성이 같은 노드들이 더 잘 연결되는지"를 포착합니다.



##### Unified ERGM Formulation

ERGM의 구조적 요인과 속성 정보(covariates)를 하나의 수식으로 표현하면 다음과 같습니다.


$$
P_{\theta, \beta}(Y = y \mid X = x) = \frac{1}{\kappa(\theta, \beta)} \exp\left\{ \theta^T g(y) + \beta^T h(y, x) \right\}
$$


이 수식은 구조적 특성과 노드 속성을 동시에 반영한 종합적인 네트워크 생성 모델로 네트워크 구조와 속성 효과의 두 요소가 같이 포함되어 있습니다.



##### Expanded Example Form


$$
P_{\theta, \beta}(Y = y \mid X = x) =  \\\frac{1}{\kappa(\theta, \beta)} \exp \Bigg\{ 
    \theta_1 S_1(y) + 
    \theta_2 \mathrm{AKT}_\lambda(y) +
    \beta_1 \sum_{i < j} y_{ij} I\{x_i = x_j\} + 
    \beta_2 \sum_{i < j} y_{ij} (x_i + x_j)
\Bigg\}
$$


이 구성 요소에서는 S(y)는 전체 엣지 수를 나타냅니다. AKT는 Alternating k-Triangle 통계량으로 삼각형 구조를 반영해줍니다. 노드 속성이 같을수록 연결될 확률이 높다는 동질성도 모델링되어 있고, 속성 값의 수준이 연결 확률에 영향을 미치는 main effect도 함께 모델링 되어 있습니다.

exp() 부분의 첫번째와 두번째 항에서는 그래프 구조를 포착하는 항으로 이루어져있고, 세번째와 네번째 항에서는 구조와 상관없이 vertex의 특성을 추가한 항으로 이루어져있습니다.

```R
> lazega.ergm <- formula(lazega.s ˜ edges +
     gwesp(log(3), fixed=TRUE) +
     nodemain("Seniority") +
     nodemain("Practice") +
     match("Practice") +
     match("Gender") +
     match("Office"))
```

`nodemain("Seniority")`와 `nodemain("Practice")`의 경우 Main effect로 포착되고, 

`match("Practice")`, `match("Gender")`, `match("Office")`는 Homophily effect로 포착됩니다.





##### Model fitting

ERGM은 exponential family에 속하는 모델로 MLE를 사용할 수 있는데, ERGM의 경우 네트워크 전체 구조를 고려해야하고, 가능한 모든 그래프에 대해 확률을 합산하는 정규화 상수 계산이 필요해서 일반적인 모델보다는 복잡하다는 단점이 있습니다.  

간단하게 수학적으로 나타내면, 


$$
P_{\theta}(Y = y) = \frac{1}{\kappa(\theta)} \exp\left( \theta^T g(y) \right)
$$


와 같은 형태로 exponential family의 전형적인 구조로 나타납니다.


$$
\ell(\theta) = \theta^T g(y) - \psi(\theta) \quad \text{where} \quad \psi(\theta) = \log \kappa(\theta)
$$
MLE (최대우도추정)은 로그우도함수에서 theta를 최대화하는 값을 찾는 것인데, 계산하는데 복잡한 과정이 필요합니다. 

그래서 우리는 R 패키지를 이용하여 Model fitting을 진행하게 됩니다.



```R
> A <- get.adjacency(lazega)
> v.attrs <- as_data_frame(lazega, what = "vertices")
> 
> lazega.s <- as.network(as.matrix(A), directed = FALSE)
> 
> set.vertex.attribute(lazega.s, "Office", v.attrs$Office)
> set.vertex.attribute(lazega.s, "Practice", v.attrs$Practice)
> set.vertex.attribute(lazega.s, "Gender", v.attrs$Gender)
> set.vertex.attribute(lazega.s, "Seniority", v.attrs$Seniority)
> 
> lazega.ergm <- formula(lazega.s ~ 
+                          edges +
+                          gwesp(log(3), fixed = TRUE) +
+                          nodemain("Seniority") +
+                          nodemain("Practice") +
+                          match("Practice") +
+                          match("Gender") +
+                          match("Office")
+ )
> 
> lazega.ergm.fit <- ergm(lazega.ergm)
> anova(lazega.ergm.fit)
Analysis of Deviance Table

Model 1: lazega.s ~ edges + gwesp(log(3), fixed = TRUE) + nodemain("Seniority") + 
    nodemain("Practice") + match("Practice") + match("Gender") + 
    match("Office")
         Df Deviance Resid. Df Resid. Dev Pr(>|Chisq|)    
NULL                       630     873.37                 
Model 1:  7   414.24       623     459.13    < 2.2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
```



이 예시는 Lazega 변호사 네트워크 데이터셋을 기반으로 ERGM을 적합하고 이 모델에 대해 ANOVA 분석을 수행하여 유의미성을 평가하는 분석 코드입니다.

anova(lazega.ergm.fit) 결과 코드를 보면, NULL에 비해 Model 1의 Residual Deviance (잔차 이탈도)가 급격히 줄어든 경우, 모델이 데이터에 대해 설명을 잘한다는 의미를 가집니다. 



ANOVA 분석처럼 ERGM과 GLM도 비슷한 구조를 공유합니다.

```R
> summary(lazega.ergm.fit)
Call:
ergm(formula = lazega.ergm)

Monte Carlo Maximum Likelihood Results:

                              Estimate Std. Error MCMC % z value Pr(>|z|)    
edges                        -6.994522   0.682010      0 -10.256  < 1e-04 ***
gwesp.fixed.1.09861228866811  0.591354   0.087768      0   6.738  < 1e-04 ***
nodecov.Seniority             0.024714   0.006305      0   3.919  < 1e-04 ***
nodecov.Practice              0.395820   0.108429      0   3.651 0.000262 ***
nodematch.Practice            0.767632   0.196495      0   3.907  < 1e-04 ***
nodematch.Gender              0.725628   0.250477      0   2.897 0.003768 ** 
nodematch.Office              1.157722   0.198807      0   5.823  < 1e-04 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

     Null Deviance: 873.4  on 630  degrees of freedom
 Residual Deviance: 459.1  on 623  degrees of freedom
 
AIC: 473.1  BIC: 504.2  (Smaller is better. MC Std. Err. = 0.3235)
```



이 결과는 lazega 데이터를 ERGM 모델에 대해 학습하고 요약한 결과인데, `nodecov.Seniority`와 `nodecov.Practice`는 main effect에 해당하는 항목입니다. homophily effect는 `nodematch.Practice`와 `nodematch.Gender`, `nodematch.Office`에 해당합니다.

homophily effect 에서는 비슷한 속성끼리 연결되는 경우를 포착하는 것인데, `nodematch.Practice`에서는 기업과 기업, 송사와 송사 형태로 matching될 경우 포착하는 효과입니다.



그렇다면, ERGM의 계수는 어떻게 해석할 수 있는지 확인해봅시다!


$$
\log \left( 
\frac{
P_{\theta}(Y_{ij} = 0 \mid Y_{(-ij)} = y_{(-ij)})
}{
P_{\theta}(Y_{ij} = 1 \mid Y_{(-ij)} = y_{(-ij)})
}
\right)
= \theta^T \Delta_{ij}(y) \\
\text{where} \ \ \Delta_{ij}(y) \ \ \text{the change statistic for the edge}\ ij \\
\Delta_{ij}(y)=g(y^+)-g(y^-)
$$


엣지 ij가 생길 확률은 나머지 네트워크가 고정되어 있을 때, ij 엣지를 추가했을 때 모델 통계량이 얼마나 변하느냐에 따라 결정됩니다.



그러면 우리가 도출한 R코드를 통해 어떻게 해석할 수 있는지 알아봅시다.

ERGM 계수를 해석하는 방법으로는 log-odds → odds ratio 변환을 통해 의미를 부여하는 방식인데, 


$$
\text{Odds Ratio = exp}(\theta)
$$


을 계산해야 실제 연결 확률 변화의 효과를 이해할 수 있게 됩니다.



먼저 Practice 계수를 해석해봅시다.


$$
\log \left( 
\frac{
P_{\theta}(Y_{ij} = 0 \mid Y_{(-ij)} = y_{(-ij)})
}{
P_{\theta}(Y_{ij} = 1 \mid Y_{(-ij)} = y_{(-ij)})
}
\right)_{\text{Practice}}
= \theta_{\text{Practice}} \cdot \left( I_{\{x_i = \text{corporate}\}} + I_{\{x_j = \text{corporate}\}} \right) \\ \\
\text{Practice : the type of vertex attribute} 
$$

$$
\theta_{\text{practice}}=0.3958 \\
\text{odds ratio = exp(0.3958)} \approx1.485
$$


practice에서 1에 해당하는 corporate에 속한 사람은 0에 해당하는 litigation에 속한 사람보다 연결될 odds가 약 1.485배, 약 48.5% 더 높은 협업의 가능성이 존재한다고 해석할 수 있습니다.



**삼각형 효과 (transitivity)**도 포착할 수 있는데, 이는 `gwesp.fixed.1.09861228866811` 를 통해서 계수를 추정하고, 유의미한지 확인할 수 있습니다. transitivity가 연결을 촉진하는 방향은 0.5876의 계수를 가지고 있어 **양의 효과**를 포착합니다. 또, p-value를 통해 0.1%의 유의수준에서 유의한 결과를 나타낸다는 것을 알 수 있습니다. 즉, 이 모델에서 transitivity는 실제 네트워크 형성에 중요한 구조적 유인이라는 것을 확인할 수 있습니다.


$$
y_{ij} \, h(x_i, x_j) = y_{ij} \left( x_i + x_j \right)
$$

$$
\Delta_{ij} = \left( x_i + x_j \right)
$$

corporate practice(기업 법무 직군)에 대한 계수 0.3954를 오즈 비(odds ratio)로 변환하면 
$$
\exp⁡(0.3954)\approx1.485
$$
이는 litigation(소송) 직군에 비해 약 48.5% 더 높은 협업 확률을 나타냅니다.

또한, 삼각형과 관련된 항목 (transitivity)도 속성 기반 동질성(homophily)을 고려한 이후에도 여전히 유의미하게 양의 영향을 주는 것으로 나타났습니다. → 이는 사회적 과정 속에 내재된 구조적 연결 경향성을 보여줍니다.

또한, ERGM에서 나타나는 p-value는 전통적인 통계검정에서의 p-value와 달라 엄밀한 확률 해석을 하기 보다는 해석을 돕는 참고 지표입니다.



```R
> gof_lazega.ergm <- gof(lazega.ergm.fit)
> par(mfrow=c(2,2))
> plot(gof_lazega.ergm)
```



<img src="{{site.url}}\images\2025-05-06-network_chap6_1\gof_ergm.PNG" alt="gof_ergm" style="zoom: 50%;" />



Goodness of fit의 결과를 통해 이 ERGM 모델은 현실 네트워크의 구조적 특성과 속성 기반 패턴을 잘 설명하고 있다는 것을 확인할 수 있습니다.





