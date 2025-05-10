---
published: true
layout: single
title:  "Network : Mathematical Models for Network Graphs"
categories: Network
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Mathematical Models for Network Graphs'**에 대한 내용을 담고 있습니다.

---



### Chapter 5 Mathematical Models for Network Graphs

#### Introduction

Network Graph Model은 다음과 같이 정의할 수 있습니다.


$$
\{ P_\theta(G),\ G \in \mathcal{G} : \theta \in \Theta \} \\
\mathcal{G} : \text{collection of ensemble of graphs} \\
P_\theta : \text{probablity distribution over} \ \ \mathcal{G} \\
\theta \in \Theta : \text{vector of parameters}
$$


관찰하고 있는 그래프 G는 theta라는 파라미터에 따라 결정되는 확률분포에 의해 생성된다고 봅니다.

이 모델을 사용하는 목적은 구조적 속성의 유의성을 평가하고, 실증적 현상 생성 메커니즘에 대한 모델링, 정점 간의 연결을 예측하는데 쓰입니다. 

모델은 두 가지 유형으로 나뉩니다. 먼저, Mathematical models,  수학적으로 다루기 쉽지만, 데이터 기반 모델은 아닙니다(예: Erdős–Rényi 모델). 두번째는 Statistical models, 데이터를 잘 맞추기 위해 설계되었으나, 수학적으로는 분석하기 어려울 수 있습니다.



#### Classical Random Graph Models

##### Types of Classical Random Graph Model

**Erdős–Rényi Model**

Erdős–Rényi Model은 Random graph model을 그래프 집합 G에 대한 Uniform Distribution로 정의합니다. 


$$
G_{N_v,N_e} \\
N_v : \text{number of vertices} \\
N_e : \text{number of edges in model}
$$


이 모델에서는 모든 가능한 N_v개의 정점 위에 정확히 N_e개의 간선을 갖는 그래프들을 동일한 확률로 선택합니다. 그 확률은 다음과 같습니다.


$$
P(G) = \binom{N}{N_e}^{-1}, \quad \text{where } N = \binom{N_v}{2}
$$


즉, N_v개의 정점이 있을 때, 가능한 모든 간선의 쌍은 위에서 정의한 N개이고, 이 중 N_e개의 간선을 무작위로 선택하는 방식입니다. 가능한 모든 그래프들 중 하나를 균등한 확률로 고르는 모델입니다.

정리하면, 정점 수와 간선 수는 고정한 상태에서, 그 간선들이 어디에 있을지 랜덤하게 결정하는 모델이고, 이 모델 하에서는 동일한 정점 수와 간선 수를 가지는 그래프들 사이에 차별이 없다, 즉 모든 그래프가 똑같은 확률로 선택된다는 의미를 가지고 있습니다. 

예를 들어, 

vertex의 개수(N_v)가 4개라면, 가능한 edge의 수는 6개입니다. 즉, 4개의 정점 사이에는 총 6개의 서로 다른 간선을 만들 수 있습니다. 만약, 딱 3개의 간선만 가지는 그래프를 원한다(N_e)면, 

가능한 간선 6개 중에서 3개를 고르는 모든 방법의 수는 


$$
\binom{6}{3} = 20
$$


이 됩니다. 총 20개의 서로 다른 그래프가 가능하고, 각각의 그래프는 모두 1/20의 확률이 나올 수 있습니다. 즉, 일정한 정점의 개수를 가진 상태에서 가능한 간선의 개수를 선택하여 만든 그래프 하나하나가 모두 동일한 확률로 생성됩니다.



**Bernoulli (Gilbert) Model**


$$
G_{N_{v,p}} \\
p \in (0,1)
$$


각 정점쌍 (i,j)에 대해 독립적으로 간선을 추가할지 말지 결정하는 방식으로, 간선을 정확히 고정하지 않고 확률적으로 생성됩니다. 간선이 생길 확률은 0~1로 설정됩니다.



두 모델의 관계는 다음과 같습니다.
$$
p_e \approx {N_E}/ {\binom{N_v}{2}}
$$
로 설정하면, 두 모델은 점근적으로 동일한 결과를 만들어냅니다. N_v가 무한히 커질 때와 같은 통계적 특성을 가집니다.



##### Classical Random Graph Models : Example and Observations

<img src="{{site.url}}\images\2025-04-21-network_chap5_1\CRG.PNG" alt="CRG" style="zoom:50%;" />



```R
> g.er <- erdos.renyi.game(100, 0.02)  
> plot(g.er, layout=layout.circle, vertex.label=NA)
> is_connected(g.er)
[1] FALSE
> 
> table(sapply(decompose(g.er), vcount))

 1  2  3 77 
18  1  1  1 
```

총 100개의 정점이 있고, 간선 생성 확률을 0.02라고 할 때, 이 값으로 생성된 그래프는 연결되어있지 않습니다. 즉,  not connected 상태입니다.  또, 정점 수가 77개인 giant component가 존재하고, 나머지는 작은 isolated component들로 구성되어 있습니다.

만약, 랜덤 그래프에서 Giant component가 생기려면,


$$
p=\frac{c}{N_v} \ \ \ \& \ \ \ c>1
$$

$$
\deg(v) \sim \text{Binomial}(N_V - 1,\ p)\ \approx\ \text{Poisson}(\lambda)
$$



c가 1보다 크면, giant component가 발생할 확률이 커집니다. 계속해서 child 노드의 개수가 1개 초과하면서 이어지기 때문에, giant component가 만들어집니다. 만약 c가 딱 1이라면, giant component는 발생하지 않고, 1보다 작다면, 어느순간 연결이 끊어지게 됩니다.


$$
\frac{c}{N_v} \cdot (N_v-1) \approx c
$$



위의 네트워크 그래프를 R코드를 이용하여 실험에서의 평균 차수를 계산해보면, 다음과 같습니다.

```R
> mean(degree(g.er))
[1] 1.86
```

$$
\lambda = p \cdot (N_V - 1) = 0.02 \cdot 99 = 1.98
$$

실제로도 평균 차수는 1.98로 기대값과 유사한 값을 나타냈습니다.

```R
> hist(degree(g.er), col="lightblue", xlab="Degree", ylab="Frequency")
```

<img src="{{site.url}}\images\2025-04-21-network_chap5_1\possion.PNG" alt="possion" style="zoom:50%;" />

이 결과는 대략적으로 포아송 분포처럼 보입니다. 특히 vertex의 개수가 크고 p가 작을 때 이항분포가 포아송 근사로 수렴하는 것과 비슷한 형태를 가집니다.



##### Classical Random Graph Models : Diameter

```R
> mean_distance(g.er) # average.path.length
[1] 6.262116
> diameter(g.er)
[1] 14
```

mean_distance (=average.path.length) : 모든 정점 쌍 사이의 최단 거리의 평균

diameter : 그래프 상의 가장 먼 두 정점 사이의 최단 거리 

diameter는 전체 vertex에서 shortest.path가 14보다 작거나 같다는 의미를 가집니다. 이는 **short path로 판단합니다.**

네트워크의 정점 수가 커질수록, 그래프의 diameter도 커지는데, 이는 선형적으로 증가하지 않고, 천천히 로그 스케일로 증가합니다. vertex의 개수에 비해 diameter는 빠르게 늘어나지 않습니다.



path가 짧고 diameter가 작은 것에 대한 이론적인 부분을 다뤄보면,

각 노드가 평균적으로 c개의 이웃을 가진다고 가정합니다.

한 정점에서 출발하면, 연결 가능한 정점의 수가 기하급수적으로 증가합니다.


$$
1,c,c^2,c^3, \cdots,c^k
$$


전체 노드에 도달하기 위해 필요한 단계의 수 k는 대략


$$
c^k \approx N_v \ \  \rightarrow \ \ k \approx log_c(N_v)
$$


각 노드가 이웃을 "균등 확률로 발생"시키는 구조는 각 노드가 child를 만드는 과정과 통계적으로 동일, 따라서 k 단계 후 도달 노드 수가 기하급수적으로 성장한다는건 노드생성 조건이 이웃생성조건과 구조적으로 같음

또 노드 간 독립성이 존재, 각 노드의 이웃 수가 이항 분포 또는 포아송 분포를 따르기 때문에 노드 수가 커질수록 중심극한정리에 따라 편차가 작아짐.



즉, 한 정점에서 나머지 정점에 도달하기 위해 걸리는 평균 거리는 다음과 같습니다.


$$
\mathbb{E}[\text{path length}] = O(\log N_V)
$$


→ diameter가 커지는 속도는, N_v가 linear하게 커지면, diameter는 log linear하게 커집니다.



##### Classical Random Graph Models : clustering

Clustering coefficient는 한 노드의 이웃끼리 서로 연결되어 있을 확률로, 노드 v가 이웃 i,j가 서로도 연결되어 있는지 보는 지표입니다. 

clustering : 그래프가 얼마나 잘 뭉쳐져(연결되어) 있는지를 판단하는 계수 _ **transitivity**

```R
> transitivity(g.er)
[1] 0
```

transitivity 계수가 0으로 작은 값을 나타냅니다.



ER 그래프에서는, edge들은 완전히 독립적으로 생성되므로,


$$
\mathbb{P}(i \sim j \mid i \sim v,\ j \sim v) = p
$$


즉, v의 두 이웃이 연결될 확률도 그냥 p가 됩니다.

<img src="{{site.url}}\images\2025-04-21-network_chap5_1\transitivity.png" alt="transitivity" style="zoom:50%;" />

edge들이 독립적으로 생성되기 때문에, condition의 유무 상관없이 p의 확률을 가지게 됩니다.



$$
p=\frac{c}{N_v} \ \ \rightarrow \ \ C=O\ (\frac{1}{N_v})
$$


일반적으로 위와 같이 정의하면, Classical Random Graphs에서는 노드 수가 많아질수록 클러스트링 계수는 0에 수렴합니다.



#### Generalized Random Graph Models

Classical Random Graph 모델을 더 현실적인 특성에 맞게 확장한 모델들을 말합니다. 기존 모델은 가능한 모든 그래프 중에서 균등하게 또는 확률적으로 생성하고, 구조적 제약이 거의 없었습니다. 그런데, 일반화한 모델에서는 그래프 G가 특정한 특성을 만족하도록 제한한 모델입니다.



##### Generalized Random Graph Models : Fixed Degree Sequence

이 모델은 그래프 집합이 정점의 차수가 고정된 그래프들로 이루어진 집합입니다.

앞의 고전적 랜덤 그래프는 정점의 차수가 다 달랐는데, 이번에는 고정되어있습니다. 또, 그래프 집합 내의 그래프들에 동일한 확률을 부여합니다.



모든 그래프는 같은 edge의 수를 가지고 있습니다.


$$
\bar d=\frac{2N_E}{N_V}
$$


전체 degree의 합은 edge의 수에 2를 곱한 값이 됩니다. 그러나, 클러스터링, diameter 등 구조적 특성은 달라질 수 있습니다. 같은 차수 구조라도 연결 방식이 다르면, 그래프 전체 구조는 완전히 달라지게 됩니다.



```R
> degs <- c(2,2,2,2,3,3,3,3)
> g1 <- sample_degseq(degs, method="vl")
> g2 <- sample_degseq(degs, method="vl")
> graph.isomorphic(g1, g2)
[1] FALSE
```



![GPH]({{site.url}}\images\2025-04-21-network_chap5_1\GPH.PNG)

두 그래프가 구조적으로 동일한지 비교했을 때, 결과가 FALSE가 나온 것으로 보아, 같은 degree sequence라도 연결방식이 다르다는 것을 확인할 수 있었습니다.



아래의 R코드는 실제 네트워크(yeast 데이터)와 같은 degree sequence를 가지는 랜덤 그래프를 비교하고 있습니다. 이는 Fixed Degree Sequence Model의 한계도 함께 나타내고 있습니다.

```R
> degs <- degree(yeast)
> fake.yeast <- degree.sequence.game(degs, method="vl")
> all(degree(yeast) == degree(fake.yeast))
[1] TRUE
 
> diameter(yeast)
[1] 15
> diameter(fake.yeast)
[1] 8

> transitivity(yeast)
[1] 0.4686178
> transitivity(fake.yeast)
[1] 0.03904827
```



**yeast vs fake.yeast**

Diameter : 15 vs 8

Clustering : 0.47 vs 0.04

같은 degree라고 해서 구조도 같진 않다는 것을 보여주고 있습니다. Fixed Degree Sequence Model은 단지 각 노드가 몇 개의 이웃을 가지는지만 고정하지, 누구와 연결되는지는 전혀 통제하지 않고 있습니다. 이는 현실성이 떨어질 수 있습니다.



#### Network Graph Models Based on Mechanisms

"현대의 네트워크 모델링에서는, 기존의 단순한 무작위 그래프에서 벗어나, 현실 시계의 특성을 반영하는 간단한 메커니즘 기반 모델로 전환된 것이 중요한 혁신이었다."

→ 지금은 어떻게 연결이 생기는지를 설계하여 현실적인 네트워크를 모델링 함 !!!



##### Small-World Models

현실 네트워크들은 보통 아래 두 가지 주요 특성을 가지고 있습니다.

High average clustering coefficient

Short Average shortest path length : classical random graph models의 특징



그리고 두 가지 특성을 잘 만족하는 가장 대표적인 모델은 Watts-Strogatz (WS) 모델입니다.

Classical Random Graph는 경로는 짧지만, 클러스터링은 거의 없습니다. 격자 구조의 경우 클러스터링은 높지만, 경로가 긴데, WS 모델은 이 두 가지 장점을 동시에 갖는 중간 구조를 제공합니다.

즉, Watts–Strogatz 모델은 높은 클러스터링과 짧은 경로 길이를 동시에 갖는 현실적인 네트워크를 생성하기 위해 고안된 small-world 모델입니다.



##### WS Model

N_v개의 노드를 원형(ring) 격자로 배치합니다. 

각 노드는 양옆 r개의 이웃과 연결합니다.

각 간선을 확률 p로 무작위 재배선을 합니다. 연결 대상 노드를 무작위로 바꾸고 자기 자신이나 중복 간선은 피하게 합니다.

이러한 메커니즘으로, 높은 클러스터링을 유지하면서 경로 길이를 줄일 수 있습니다.



이러한 과정은 본래 규칙적인 네트워크(구조화된 네트워크)에 무작위성을 도입하면서, 평균 경로 길이를 줄이고, local clustering은 상당히 높은 수준으로 유지하게 합니다. p가 아주 작아도, 경로 길이에 극적인 감소를 일으키기에 충분합니다.

<img src="{{site.url}}\images\2025-04-21-network_chap5_1\WS.PNG" alt="WS" style="zoom: 67%;" />

먼저, g.lat100의 경우, edge가 전혀 재배선되지 않은 상태로 완전한 규칙 격자 구조 형태를 가지고 있습니다. (즉, randomness=0)

clustering 계수의 경우 0.667 (매우 높고 이웃끼리도 잘 연결됨)

DIameter는 10 (네트워크에서 가장 먼 두 노드의 거리)

Average path length는 5.45 (평균 거리 길다!)



→ 정보 확산이 느리고, 지역적 연결은 좋지만, 전체 연결 효율은 낮습니다.



반면에, p=0.05로 작은 확률로 재배선하는 경우, 전체 구조는 여전히 꽤 지역적이지만, 몇몇 "지름길(shortcuts)"이 생깁니다.

clustering 계수의 경우 0.4864 (여전히 높고, 지역성이 유지됩니다.)

DIameter는 4 (급격히 줄어들어 더 빠르게 도달 가능합니다.)

Average path length는 2.67 (절반 이하로  줄어듭니다.)

→ 아주 작은 재배선만으로도 네트워크 전반의 효율성이 급상승하고 local 구조와 global 연결성을 모두 만족하는 small-world network 가 생기게 됩니다.

<img src="{{site.url}}\images\2025-04-21-network_chap5_1\wsplot.PNG" alt="wsplot" style="zoom:80%;" />

빨간 선 : diameter

파란 선 : clustering coefficient normalize : 로그 형태로 급격하게 떨어지진 않는다.



##### Preferential Attachment Models

인터넷, 논문 인용망, SNS와 같은 현실 네트워크에서는 일부 노드는 엄청나게 많은 연결을 가지고, 대부분의 노드는 연결 수가 적습니다. 이러한 구조는 단순한 Random Graph로 설명하기에는 한계가 있습니다.

그래서, Barabási–Albert (BA) 모델과 같은 모델들은 네트워크의 성장과 선호적 연결을 잘 포착합니다. 이러한 모델은 소수의 노드가 대부분의 노드보다 훨씬 많은 연결을 가지는 현상을 효과적으로 설명해내는 특징을 가지고 있습니다. BA 모델의 핵심 아이디어는, 새로 들어오는 노드일수록, 이미 연결이 많은 노드에 더 연결되려는 경향이 있습니다. 이로 인해 **부익부 현상(rich-get-richer)**이 발생하게 됩니다.



이 모델의 메커니즘은 다음과 같습니다.

초기에는 작은 네트워크에서 시작합니다. 처음에는 작은 개수의 노드와 간선으로 시작해서, 매 시간 t마다 새로운 노드를 하나 추가하고, 기존 노드 중 m개를 골라서 간선을 연결시킵니다.

물론, 연결하는 그 대상은 랜덤이 아니고, 연결이 많은 노드에게 더 높은 확률로 연결됩니다.


$$
\text{Probability proportional to degree} = \frac{d_v}{\sum_{v'}d_{v'}}
$$


여기서 분모는 총 vertex의 degree합을 의미합니다. 분자는 노드 v의 현재 연결 수(degree)를 의미합니다. 즉, 연결이 많은 노드일수록 선택될 확률이 높습니다. 

시간이 지나면, 몇몇 노드는 점점 더 많은 연결을 가지게 되고 이는 hub를 형성할 수 있게 됩니다. 대부분의 노드는 연결이 적고, 소수만 연결이 많아서, 이를 skewed degree distribution이라고 합니다. (한쪽으로만 몰린!)

```R
> set.seed(42)
> g.ba <-sample_pa(100,directed=FALSE)
> plot(g.ba,layout=layout.circle,vertex.label=NA)
> hist(degree(g.ba),col="lightblue")
> summary(degree(g.ba))
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   1.00    1.00    1.00    1.98    2.00    9.00 
```



Preferential Attatchment (선호 연결)을 기반으로 노드 100개짜리 BA 그래프를 생성한 후, 원형 레이아웃으로 시각화합니다. 이를 통해 도출된 노드의 degree를 히스토그램으로 시각화했을 때, 다음과 같은 히스토그램이 나타납니다.

<img src="{{site.url}}\images\2025-04-21-network_chap5_1\BA.PNG" alt="BA" style="zoom:67%;" />

각 노드의 degree 요약 통계를 출력했을 때, max degree = 9라는 결과는 중요한 시사점을 줍니다.

대부분의 노드 degree는 작은데, 소수 노드는 9개 연결을 갖는 허브(hub) 역할을 합니다.

**많은 노드는 연결이 적고 일부만 아주 많은 연결을 갖는다는 특성을 가지고 있습니다.**

```R
> mean_distance(g.ba)
[1] 5.815556
> diameter(g.ba) 
[1] 12
> transitivity(g.ba)
[1] 0
```



BA 모델에서는 대부분의 노드는 연결이 적고, 일부 노드는 매우 많은 연결 (hub)을 가집니다. 연결 수의 분포가 불균형적이고 편향되어 있습니다. 또, 연결 수가 거듭제곱 법칙(power-law)을 따릅니다. 확률적으로 degree d를 가진 노드의 비율은 


$$
d^{-3}
$$


로, degree가 커질수록 빠르게 드물어지는 경향이 있습니다. 이를 긴 꼬리 분포라고 부릅니다. (대다수의 노드는 연결이 적고, 적은 노드는 많이 연결되어 있습니다.)

무작위로 노드를 제거해도 대부분은 low-degree여서 전체적인 구조는 유지되지만, 만약 hub를 표적 제거하면 네트워크가 빠르게 분해됩니다. 현실 네트워크의 취약성과 회복력 분석에 유용합니다. 

BA 모델은 구조적으로 삼각형이 잘 생기지 않아서 클러스터링 계수가 낮습니다. 그리고, 경로는 hub를 통해 빠르게 연결되기 때문에 평균 거리(average path length)는 작습니다. 



#### Assessing Significance of Network Graph Characteristics

Erdős–Rényi, Watts–Strogatz, Barabási–Albert와 같은 네트워크 그래프 모델들은 현실 세계의 복잡한 네트워크 구조를 완벽히 설명하기에는 지나치게 단순합니다. 그럼에도 불구하고, 이 모델들은 **통계적 가설 검정(statistical hypothesis testing)**에서 중요한 역할을 하는데, 특히 **관측된 네트워크의 특정 특성이 우연히 나타난 것인지, 아니면 의미 있는 구조적 특징인지**를 평가할 때, 이러한 모델들이 기준선(reference model)으로 사용됩니다. 즉, 단순 모델을 바탕으로 만들어진 랜덤 그래프들과 비교함으로써, 실제 네트워크에서 관측된 클러스터링, 중심성, 경로 길이 등의 값이 **통계적으로 유의미한지**를 판단할 수 있게 됩니다.



##### General Testing Framework

네트워크 통계량의 유의성 평가

우리가 어떤 실제 네트워크 G_obs를 관측한다고 할 때, 이때, 관심 있는 구조적 속성을 다음과 같다고 표현해봅시다.


$$
\text{real network} \  : \ G_{obs} \\
\text{structural property} \ : \ \eta(G)
$$

$$
\text{To assess whether } \eta(G_{obs}) \ \text{is unusually large or small}, \\
\text{we generate a reference distribution under a null model:} \\
$$

$$
P_{\eta, \mathcal{G}}(t) = \frac{ \# \{ G \in \mathcal{G} : \eta(G) \le t \} }{ |\mathcal{G}| }
$$



만약 structural property가 t이하인 그래프의 비율을 의미하며, 귀무 분포에서의 누적 분포 함수 역할을 합니다. structural property 값이 분포의 극단적인 꼬리 부분에 위치한다면, 실제 네트워크가 해당 귀무 모형에서 생성됐을 가능성이 매우 낮다는 증거가 됩니다. 이 원리는 Motif detection에서도 활용됩니다. 실제 네트워크에서는 특정 작은 subgraph들이 무작위 네트워크보다 훨씬 자주 등장하는 현상을 보이며 이는 통계적으로 유의하다고 판단합니다.



실제 네트워크 데이터를 이용해서 관측 네트워크에서 커뮤니티 수가 과연 "우연히" 생긴 구조인지 귀무 모형을 설정해서 비교해봤습니다.

먼저 Karate Club 네트워크를 기반으로 관측된 커뮤니티 개수(3개)가 무작위 네트워크에서 기대되는 값과 비교하여 유의미하는지 검정하는 과정인데,

Erdős–Rényi 모델은 같은 노드 수 (34개), 같은 간선 수 (78개)가 존재하고, 무작위로 간선을 연결합니다.

Degree-preserving 모델은 원래 네트워크의 degree-sequence는 유지하면서 간선을 무작위로 재배치합니다.



```R
> nv <- vcount(karate)     
> ne <- ecount(karate)     
> degs <- degree(karate)   
> ntrials <- 1000
> num.comm.rg <- numeric(ntrials)
> num.comm.grg <- numeric(ntrials)
> 
> # ER Model
> 
> for (i in 1:ntrials) {
+   g.rg <- erdos.renyi.game(nv, ne, type="gnm")     
+   c.rg <- cluster_fast_greedy(g.rg)               
+   num.comm.rg[i] <- length(c.rg)                   
+ }
> 
> # Degree-preserving Model
> 
> for (i in 1:ntrials) {
+   g.grg <- degree.sequence.game(degs, method="vl") 
+   c.grg <- fastgreedy.community(g.grg)
+   num.comm.grg[i] <- length(c.grg)
+ }
> 
> rslts <- c(num.comm.rg,num.comm.grg)
> indx <- c(rep(0,ntrials),rep(1,ntrials))
> counts <- table(indx,rslts)/ntrials
> 
> 
> barplot(counts, beside=TRUE, col=c("blue", "red"),
+         xlab="Number of Communities", ylab="Relative Frequency",
+         legend=c("Fixed Size", "Fixed Degree Sequence"))
```

<img src="{{site.url}}\images\2025-04-21-network_chap5_1\GTF.PNG" alt="GTF" style="zoom:50%;" />

두 모델 모두에서 가장 자주 나타나는 커뮤니티 수는 5~6개이고, 3개의 커뮤니티가 나오는 경우는 거의 없습니다. barplot의 왼쪽 구석에만 3에 해당하는 조그만 막대가 존재합니다. 

Random Graph에서는 보통 5~6개의 그룹이 자동적으로 생기는데, 실제 네트워크에서는 딱 3개의 커뮤니티만 존재하고, 커뮤니티 수가 작은건 랜덤한 결과로는 거의 안 나오는 일이어서, 의미 있는 커뮤니티 구조가 존재한다고 할 수 있습니다.

Fixed size : 모델을 통해 찾은  (community 내재하지 않은 uniformly random한 모형)

Fixed degree sequence : Karate랑 같은 degree (vertex degree)

