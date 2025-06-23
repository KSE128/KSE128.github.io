---
published: true
layout: single
title:  "Network : Network Topology Inference (2)"
categories: Network
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Network Topology Inference' 의 Part 2**에 대한 내용을 담고 있습니다.

---

### Chapter 7  Network Topology Inference (Part 2)



#### Association Network Inference

데이터가 풍부한 과학 분야에서는 연관성(association)을 바탕으로 네트워크를 정의하는 경우가 많습니다. 여기서 연관성이란, 두 개체가 서로 얼마나 비슷하거나 관계가 있는지를 수치로 나타낸 것을 의미하는데요. 각 노드는 여러 개의 특성값을 가지게 됩니다. 예를 들어, 유전자는 각 실험에서의 발현량이 속성이 될 수 있습니다. 

이러한 데이터셋은 { x_1, ... , x_n }의 벡터 형태로 나타내게 됩니다. 즉, 각 노드마다 벡터 형태의 데이터를 가지고 있습니다. 두 노드 i와 j의 속성 벡터 x_i와 x_j를 비교해서 유사도를 계산합니다. 계산하는 방식에는 피어슨 상관계수, 유클리드 거리, 코사인 유사도가 있습니다. 모든 노드쌍에 대해 sim(i,j)를 계산한 뒤, 이 값이 어떤 임계값 이상이면 "연결되어 있다"고 판단하여 엣지를 생성합니다. 즉, 유사도가 충분히 높으면 연결한다고 판단합니다. 

이 추론의 핵심은 이론적으로는 두 노드 간 연관성이 있다고 가정하지만, 실제로는 각 노드의 특성만 관측되고 연관성은 직접 볼 수 없습니다. 그래서 실제 데이터에서는 연관성도 간접적으로 계산하거나 추정해야합니다. 그리하여, 노드들의 속성 정보만 가지고 어떤 노드쌍 사이에 엣지가 있을지를 추론하는게 목표입니다. 실제로 엣지 존재 여부를 관측하는 것이 아니라 속성 데이터만으로 네트워크 구조를 만들어냅니다.

sim(i,j)를 선택할 때 correlation과 partial correlation을 이용하여 계산하게 됩니다.



##### Using real dataset 'Ecoli'

실제 데이터인 Ecoli 데이터를 통해 확인해봅시다.

```R
> data(Ecoli.data)
> ls()
[1] "Ecoli.expr" "regDB.adj" 
> dim(Ecoli.expr)
[1]  40 153
> dim(regDB.adj)
[1] 153 153
```



`Ecoli.expr`는 유전자 발현 데이터를 의미합니다. 이 행렬의 행은 서로 다른 40개의 유전자 조작 실험(gene knock-out 실험), 열은 153개의 전사인자 유전자 (transcription factor gene)을 의미합니다. 행렬의 각각의 값은 로그 변환된 유전자 발현량을 의미합니다. 



`regDB.adj`는 RegulonDB에서 가져온 실제 유전자 조절 네트워크를 의미합니다. 값이 1인 곳은 실제로 조절 관계가 있는 유전자 쌍을 의미합니다. 이는 엣지의 값이나 강도, 구체적인 수치 정보는 없습니다. 그저 네트워크 구조만 있고 계량적 정보는 없다는 것을 의미합니다.

`Ecoli.expr`는 실험적으로 측정한 유전자 발현 데이터이고, `regDB.adj`는 실험 외적으로 알려진 진짜 네트워크 구조를 의미합니다.

```R
> heatmap(scale(Ecoli.expr), Rowv=NA)
```



<img src="{{site.url}}\images\2025-06-07-network_chap7_2\heatmap.png" alt="heatmap" style="zoom:80%;" />



이 코드는 유전자 발현 데이터를 heatmap으로 그렸는데, 유전자들이 클러스터링 되어 있습니다. 비슷하게 발현되는 유전자들이 가까이 묶여있습니다. 또, 비슷한 발현 패천이 있는 유전자 그룹을 시각적으로 확인할 수 있습니다. 비슷한 발현 패턴이 있는 유전자들은 같은 조절 네트워크에 있을 가능성이 높습니다. 실험 그룹 내에서도 비슷한 유전자 반응 패턴이 보이는데, 이 또한 네트워크 추정에 중요한 단서가 될 수 있습니다. Heatmap으로 유전자 발현 패턴의 유사성을 한눈에 보고, 잠재적인 유전자 네트워크 구조 추론의 단서로 활용할 수 있습니다.



```R
> g.regDB <- graph.adjacency(regDB.adj, mode="undirected")
> plot(g.regDB, vertex.size=3, vertex.label=NA)
```

<img src="{{site.url}}\images\2025-06-07-network_chap7_2\truth.PNG" alt="truth" style="zoom:67%;" />



이 네트워크는 실제로 알려진 유전자 조절 관계를 보여주는 네트워크로 추론한 결과(예측 네트워크)와 비교/평가할 때의 기준으로 사용됩니다. 즉, 나중에 네트워크 추론을 하고 나면 실제 네트워크와 얼마나 비슷하고 정답을 얼마나 맞췄는지 평가하는데 사용되는 레퍼런스 네트워크 입니다.



##### Correlation Networks

Correlation Network는 네트워크의 엣지를 상관관계에 기반해서 만들게 됩니다. 각 노드는 여러 속성을 가지고 있고, 두 노드 i와 j의 속성 벡터 X_i와 X_j 사이의 피어슨 상관계수를 계산합니다.


$$
\rho_{ij} = 
\frac{ \mathrm{Cov}(X_i, X_j) }
     { \sqrt{ \mathrm{Var}(X_i) \mathrm{Var}(X_j) } }
= \frac{ \sigma_{ij} }
       { \sqrt{ \sigma_{ii} \sigma_{jj} } }
$$


Cov(X_i, X_j) 을 두 노드의 속성 사이의 공분산을 의미하고, Var(X_i)와 Var(X_j)는 각 노드 속성의 분산을 의미합니다. 상관계수로 네트워크를 생성할 때, 상관계수가 0이 아닌 경우에만 엣지를 생성합니다. 여기서 상관계수(correlation)를 similarity measure로 사용합니다.



이러한 similarity(상관계수)에 기반하여 두 노드 i,j의 상관계수가 0이 아니면 엣지를 만듭니다. 즉, 비슷하게 행동하는 노드까리 연결해서 네트워크를 만들게 됩니다.


$$
E = \left\{ \{i, j\} \in V^{(2)} : \rho_{ij} \neq 0 \right\}
$$


E는 네트워크의 엣지 집합, V는 노드 집합 V에서 가능한 모든 2개의 노드 쌍을 의미합니다. 이를 이용하여 만들어진 네트워크를 correlation graph, covariance graph이라고 합니다.



실제 데이터가 있을 때 어떤 노드쌍이 연결(상관)되어 있는지 통계적으로 판별하는 과정을 다룬다면, 가설검정의 문제로 변환하게 됩니다.



$$
H_0:\rho_{ij}=0 \ \ \text{vs} \ \ H_1:\rho_{ij} \ne 0
$$



귀무가설 : 서로 독립적이다. & 대립가설 : 서로 연관성이 존재한다.



이 가설검정의 통계량은 피어슨 상관계수를 이용해서 계산합니다.


$$
\hat{\rho}_{ij} = \frac{ \hat{\sigma}_{ij} }{ \sqrt{ \hat{\sigma}_{ii} \hat{\sigma}_{jj} } }
$$


만약 표본상관계수가 충분히 크다(통계적으로 유의하다)면, i와 j는 연결되어 있다고 판단합니다.

그러나, 상관계수는 -1과 1 사이의 값이고, 샘플 크기가 작을수록 분산이 일정하지 않고(heteroskedastic), 정규분포도 아니어서, 통계적으로 정확한 검정을 위해 분산이 일정하고 정규분포에 가까운 값으로 바꿔주는 것이 필요합니다.



그래서 우리는 Fisher의 z-변환을 수행합니다.

상관계수에 아래 변환을 적용하면 다음과 같은 수식이 나옵니다.


$$
z_{ij} = \tanh^{-1} \left( \hat{\rho}_{ij} \right)
= \frac{1}{2} \log \left( \frac{1 + \hat{\rho}_{ij}}{1 - \hat{\rho}_{ij}} \right) \\
z_{ij} \sim N\left(0,\ \frac{1}{(n-3)}\right)
$$


이 변환을 거치면, p-value를 구하거나 임계값과 비교하는 통계적 검정이 쉬워집니다.

이번엔 p-value를 계산하는 과정을 확인해봅시다. 양측 검정을 진행하기 때문에 절댓값을 사용하고 2를 곱해줍니다.


$$
p_{ij}=2\cdot\Phi\left( -|z_{ij}| \cdot \sqrt{n-3} \right) \\
\Phi : \text{CDF of the standard normal distribution}
$$

<img src="{{site.url}}\images\2025-06-07-network_chap7_2\fisher.png" alt="fisher" style="zoom:67%;" />



이러한 hypothesis testing을 각각의 case마다 진행해야합니다. 만약 노드가 N_v개라면, 가능한 노드쌍의 개수는 노드의 개수의 제곱이 될 것입니다. 그렇다면, 상관계수 유의성 검정을 유의수준 0.05로 테스트를 1000번 하면, 실제로 아무 관계 없어도 50개쯤은 우연히 유의하다고 잘못 나올 가능성이 존재합니다. 이를 해결하기 위해 **Benjamini-Hochberg(BH)** 절차를 이용합니다. **FDR(False Discovery Rate)**은 유의하다고 판단한 것 중에 진짜로 거짓인 것의 비율을 말하는데, 각 쌍에 대해 구한 p-value를 정렬해서 순위에 따라 임계값을 조정하여 FDR이 5% 이내가 되도록 유의기준을 자동으로 조절해줍니다. BH 방법으로 FDR을 제어해서 **정말 유의미한 연결만 네트워크에 포함**시키도록 합니다.





##### R Example : Correlation Networks

```R
> mycorr <- cor(Ecoli.expr)
> z <- 0.5 * log((1 + mycorr) / (1 - mycorr))
> z.vec <- z[upper.tri(z)]
> n <- dim(Ecoli.expr)[1]
> corr.pvals <- 2 * pnorm(abs(z.vec), 0, sqrt(1 / (n - 3)), lower.tail = FALSE)
> corr.pvals.adj <- p.adjust(corr.pvals, method = "BH")
> length(corr.pvals.adj[corr.pvals.adj < 0.05])
[1] 5227
```



유의(0.05 미만)한 상관관계 쌍이 5227개라는 큰 숫자로 존재함에도 불구하고, 조심스럽게 해석해야합니다. 실제로 Q-Q plot으로 확인해보면, 표본분포와 정규분포가 잘 맞는지 시각적으로 비교할 수 있는데, Q-Q plot에서 큰 차이가 보인다면, 정규분포 가정이 잘못됩니다. 그러면 계산한 p-value, 유의성 판정, 엣지 개수 자체가 신뢰도가 떨어질 수 있게 될 수 있습니다.



데이터가 많이 존재할 때, 검정 통계량의 분포를 실제 데이터로부터 경험적으로 추정하는 경우가 있습니다. fdrtool 패키지를 써서 두 군집 혼합 분포 모델로 상관계수들의 분포를 표현합니다. 이론적으로는 상관계수 등의 검정 통계량이 정규분포를 따른다고 가정하지만, 실제 데이터는 그 가정이 잘 맞지 않을 수 있습니다. 그럴 때는 실제 데이터로부터 귀무가설(null) 하의 분포를 직접 추정하는 것이 더 현실적일 때가 많습니다.



fdrtool의 혼합 분포 모델을 다루면 다음과 같습니다.


$$
f(s) = \eta_0 f_0(s) + (1 - \eta_0) f_1(s) \\
f(s) : \text{the distribution of correlation score} \\
f_0(s): \text{the densities under the null} \\
f_1(s) : \text{the densities under the alternative} \\
\eta_0 : \text{the proportion of the true null hypotheses}
$$




이는 관측된 상관계수 분포를 두 가지 군집(클러스터)의 혼합으로 모델링하고, 이를 기반으로 각 점수가 진짜로 나타나는 것인지, 우연인지를 확률적으로 판정하게 됩니다. R의 fdrtool 패키지는 혼합 모델을 적합해서 FDR을 보정합니다.



fdrtool 방식은 정규분포 가정 대신, 실제 데이터로부터 귀무/대립가설 분포를 직접 추정해서 더 현실적이고 신뢰도 높은 다중비교 보정을 할 수 있게 해주는 모델입니다.



R코드를 이용해서 상관계수 행렬에서 유의한 상관관계를 경험적으로 찾기 위한 과정을 보여주고 있습니다.

```R
> library(fdrtool)
> mycorr.vec <- mycorr[upper.tri(mycorr)]
> fdr <- fdrtool(mycorr.vec, statistic = "correlation")
```





<img src="{{site.url}}\images\2025-06-07-network_chap7_2\fdrtool.PNG" alt="fdrtool" style="zoom: 50%;" />



이 데이터셋에서는 상관계수가 커보여도 실제로 "진짜로 유의한 상관관계"는 없거나 매우 드물다고 모델이 판정했습니다. (eta0=1) 이는, 귀무분포(상관계수가 0이다)가 모든 데이터를 설명하고 있다고 판단하고, 아무 쌍도 네트워크에 추가하지 않는 결과가 나옵니다. 즉, 진짜 상관관계가 존재함은 전혀 없고, 전부 연관이 없다는 결론이 모델에서 나타나게 되었습니다. 이는 모든 상관계수가 '우연'에 불과하다고 판정되어 네트워크에 엣지를 아무것도 추가하지 않습니다. empty graph가 추론됩니다.

이 예시는 수식이나 이론적 가정(모형, parametric model)만 맹신하지 말고, 실제 데이터가 어떻게 생겼는지도 함께 고려해야 신뢰할 수 있는 분석이 가능하다는 이야기를 하고 있습니다.





##### Partial Correlation Networks

상관관계(correlation)는 전체적인 "비슷함"을 잡아내지만, 직접적인 관계(direct)와 간전적인 관계(indirect) 관계는 구분하지 못합니다. 

아래의 그림을 확인하면, X_i와 X_j가 둘다 X_k와 강한 상관이 있을 때, 세 변수 사이의 간접연결로 이뤄지는 것인데, X_i와 X_j 사이에 상관계수가 커질 수 있게 됩니다. 간접 연결이 아닌 직접 영향만 네트워크에 나타내고 싶은데, 이때 partial correlation을 사용합니다.

<img src="{{site.url}}\images\2025-06-07-network_chap7_2\partial.png" alt="partial" style="zoom: 25%;" />




$$
S_m =\{k_1, \cdots, k_m\} \subset V \setminus \{i,j\} \\
\text{a set of } m \text{ conditioning variables}
$$


S는 i와 j를 제외한 나머지 (m개) 변수의 집합입니다.


$$
\rho_{ij \mid S_m} = \frac{ \sigma_{ij \mid S_m} }{ \sqrt{ \sigma_{ii \mid S_m} \, \sigma_{jj \mid S_m} } }
$$


σ 부분은 S를 조건부로 통제했을 때, i와 j 간의 조건부 공분산을 나타냅니다.

또, 이러한 구성에 따라 partial correlation이 있는지 없는지를 결정할 수 있게 됩니다.

partial correlation의 분모와 분자에 들어가는 값들이 partial covariance matrix에서 가져온 원소들입니다. 

partial covariance matrix는 아래와 같은 형태로 나타납니다.


$$
\Sigma_{11 \mid 2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} \\
\Sigma_{11}: \text{partial covariance matrix of } (X_i,X_j) \\
\Sigma_{22}: \text{partial covariance matrix of } (X_{S_m})
$$


partial covariance matrix에서 X와 Y를 다변량 정규분포로 가정하면, 조건부 분포도 여전히 다변량 정규분포가 됩니다. 


$$
\mathrm{Cov} \left(
\left|
\begin{pmatrix}
X_i \\
X_j \\
X_{S_m}
\end{pmatrix}
\right|
\right)
= \Sigma =
\begin{pmatrix}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{pmatrix}
$$




X가 Gaussian distribution을 따르는 특수한 경우에는 다른 계산법을 이용합니다. 부분 상관계수가 0인 경우에는 X_i와 X_j가 조건부 독립이라는 의미와 완전히 일치합니다. 부분 상관이 0이면, 나머지 변수들을 이미 다 알고 있을 때 X_i 값을 알아도 X_j에 대해 추가로 알게 되는 정보가 없다는 의미로 두 변수 사이에 직접 연결이 존재하지 않는다는 의미를 가지고 있습니다.



우선, m=0인 경우를 다루면, 이때는 partial correlation = Pearson correlation가 됩니다. 이때는 간접 영향까지 포함하게 됩니다.

만약, m=1인 경우, 한 변수로 조건화하여 계산하게 됩니다.


$$
\rho_{ij \mid k} = \frac{ \rho_{ij} - \rho_{ik} \rho_{jk} }{ \sqrt{ (1 - \rho_{ik}^2)(1 - \rho_{jk}^2) } } \\
\rho_{ij} : \text{correlation of } X_i \text{ and } X_j \\
\rho_{ik} : \text{correlation of } X_i \text{ and } X_k \\
\rho_{jk} : \text{correlation of } X_j \text{ and } X_k
$$


위의 식은 i와 j의 상관관계에서 k가 설명하는 부분을 빼고 남은 **순수한 직접 관계만**을 측정하는 것입니다. 만약, i와 j의 상관이 전부 k 때문이면, 이 상관계수는 0에 가까운 값이 됩니다.


$$
S_m =\{k_1, \cdots, k_m\} \subset V \setminus \{i,j\} \\
\text{a set of } m \text{ conditioning variables}
$$

$$
E = \left\{ \{i, j\} \in V^{(2)} : \rho_{ij \mid S_m} \neq 0,\, \forall S_m \in V^{(m)} \setminus \{i, j\} \right\}
$$


엣지를 생성하기 위해서는 두 노드 i와 j 사이에서 i, j를 제외한 모든 다른 변수 집합에 대해 부분 상관계수가 모두 0이 아니어야합니다. 즉, 어떤 변수 집합으로 조건화해도 i, j 사이의 직접 연결이 남아있어야만 엣지를 인정하게 됩니다. 간접 연결을 완전히 제거하고, 어떤 경우에도 직접적인 상관이 남아있는 쌍만 연결할 수 있게 되는 것입니다. 어떤 중간 변수를 조건부를 잡아도 상관이 사라지지 않아야 직접 연결이 '진짜'라는 의미를 가지게 됩니다.



교수님 설명 !

만약 partial correlation이 **0이 아니게 되는 subset이 “하나라도” 있으면** (즉, 일부 컨디셔닝한 경우에만 직접 연결로 보이면),   **i와 j 사이에 엣지가 있다고 생각하는 해석**도 가능합니다. 현실적으로는, 모든 subset에서 0이 아니길 요구하는 건 너무 엄격합니다.



i와 j 외의 m개의 버텍스를 골라서 컨디셔닝할 때 **어떤 vertex를 고르느냐에 따라 partial correlation 값이 바뀔 수 있**습니다. 예를 들어, 1번, 2번 vertex로 컨디셔닝할 땐 partial correlation이 0이 아니게 나오지만 5번, 6번으로 컨디셔닝할 땐 0으로 나올 수도 있습니다. 즉, **컨디셔닝 집합 선택에 따라 edge 생성 여부가 달라질 수 있**습니다.

따라서 partial correlation network의 결과는 **컨디셔닝 버텍스 집합 선택에 “민감(sensitive)”하다**는 점에 주의해야 합니다.



우리가 네트워크 분석에서 partial correlation을 써서 i, j 사이에 엣지가 있는지 판단할 때, m개 vertex(컨디셔닝 변수)를 “어떻게” 고르는지, 어떤 subset을 골랐을 때 partial correlation이 0인지 아닌지가 **결과에 큰 영향을 줍니다**. 즉, **컨디셔닝 방법에 따라 네트워크가 달라질 수 있다!**



실제 네트워크 분석에서 m개의 컨디셔닝 변수 set을 바꿀 때마다 i, j 간 partial correlation이 0이기도, 0이 아니기도 하다면 네트워크 구조 결과가 “왔다 갔다” 할 수 있습니다.

이런 점 때문에 partial correlation 기반 네트워크의 엣지 정의는 **subset 선택에 매우 민감하고**, **실제로 어떻게 subset을 정하는지(전부, 일부, heuristic 등)에 따라 네트워크 결과가 달라질 수 있다**는 점을 꼭 유의해야합니다.



edge의 존재 여부를 검정하기 위해서는 가설을 다음과 같이 수립할 수 있습니다.


$$
H_0 : \exists S_m \ \text{such that} \ \rho_{ij \mid S_m} = 0 
\quad \text{vs.} \quad
H_1 : \rho_{ij \mid S_m} \neq 0 \ \text{for all } S_m
$$


**귀무가설** 

컨디셔닝 집합 중에 하나라도 partial correlation이 0인게 있다면 

→ 어떤 경우라도 부분 상관계수가 0인 경우가 있으면 edge 없음 (직접 연결 없음)



**대립가설**

모든 컨디셔닝 집합에 대해 partial correlation이 0이 아니다.

→ 어떤 경우에도 부분 상관계수가 0이 아니면 edge가 있다. (직접 연결 존재)




$$
H_0' : \rho_{ij \mid S_m} = 0 
\quad \text{vs.} \quad
H_1' : \rho_{ij \mid S_m} \neq 0
$$


실제로는 각각의 컨디셔닝 집합에 대해 개별적으로 검정합니다.

즉, 각각의 컨디셔닝 집합을 바꿔가며 여러 번의 검정(p-value 계산)을 수행합니다.


$$
p_{ij, \max} = \max_{S_m} p_{ij \mid S_m}
$$


여러 개의 p-value가 나오므로, 그 중 가장 큰 p-value를 대표값으로 사용합니다. 



##### R example : Partial Correlation Networks

Fisher 변환을 적용하여 각 노드쌍에 대해 i,j를 제외한 모든 partial correlation을 계산할 수 있습니다. 그중, 최대 p-value를 edge의 대표값으로 사용할 수 있습니다.

```R
> pcorr.pvals <- matrix(0, 153, 153)
> for (i in 1:153) {
+   for (j in 1:153) {
+     if (i != j) {
+       idx <- setdiff(1:153, c(i, j))
+       rowi <- mycorr[i, idx]
+       rowj <- mycorr[j, idx]
+       # partial correlation vector (length 151)
+       tmp <- (mycorr[i, j] - rowi * rowj) / sqrt((1 - rowi^2) * (1 - rowj^2))
+       tmp.zvals <- 0.5 * log((1 + tmp) / (1 - tmp))
+       tmp.s.zvals <- sqrt(n - 4) * tmp.zvals
+       tmp.pvals <- 2 * pnorm(abs(tmp.s.zvals), 0, 1, lower.tail = FALSE)
+       pcorr.pvals[i, j] <- max(tmp.pvals, na.rm = TRUE)
+     }
+   }
+ }
> pcorr.pvals.vec <- pcorr.pvals[lower.tri(pcorr.pvals)]
> pcorr.pvals.adj <- p.adjust(pcorr.pvals.vec, method = "BH")
> pcorr.edges <- (pcorr.pvals.adj < 0.05)
> length(pcorr.pvals.adj[pcorr.edges])
[1] 25
> sum(pcorr.pvals.vec < 0.05)
[1] 294
```



여기서 가능한 k의 수는 전체 노드의 개수에서 i와 j 두 노드를 제외한 나머지의 개수와 같습니다.

또, R 코드에서 p.adjust를 통해 모종의 보정(correction)을 통해 edge의 수가 원래 294개에서 correction을 통해 25개로 감소했음을 알 수 있습니다.



```R
> str(intersection(g.regDB, g.pcorr, byname = FALSE))
```

이 코드를 통해 실제 네트워크와 partial correlation 네트워크가 동시에 가진 엣지의 개수를 알 수 있게 됩니다.



또한, partial correlation을 이용해 얻은 p-value로 fdrtool을 바로 적용(FDR 보정)하여 edge를 판정할 수 있습니다.

아래의 식을 통해 edge를 판정할 수 있습니다.


$$
f(s) = \eta_0 f_0(s) + (1 - \eta_0) f_1(s) \\
$$


```R
> fdr <- fdrtool(pcorr.pvals.vec, statistic = "pvalue", plot = FALSE)
Step 1... determine cutoff point
Step 2... estimate parameters of null distribution and eta0
Step 3... compute p-values and estimate empirical PDF/CDF
Step 4... compute q-values and local fdr

> pcorr.edges.2 <- (fdr$qval < 0.05)
> length(fdr$qval[pcorr.edges.2])
[1] 25
```



fdrtool은 p-value의 경험적 분포로부터 mixture model을 적합해 FDR(q-value)을 산출합니다.

q-value < 0.05인 edge가 25개임을 알 수 있습니다.

```R
> length(pcorr.pvals.adj[pcorr.edges])
[1] 25
```



앞에서 다룬 BH보정(first approach)과 fdrtool로 보정한 결과가 동일하므로 **결과가 견고함을 시사**합니다. **(robustness)**





Correlation의 경우 선형성만 확인할 수 있는 도구입니다. Partial Correlation도 correlation의 variation이기 때문에 선형적 관계만 capture할 수 있습니다. 즉, 비선형성은 capture하지 못합니다.



##### Gaussian Graphical Model Networks

Gaussian Graphical Model Networks은 partial correlation의 특수한 경우입니다.

Partial Correlation 분석에서 특정 노드 쌍(i,j)에 대해 **나머지 모든 변수를 다** 조건부로 잡는 것을 가우시안 그래프 모델(Gaussian graphical model, GGM)이라고 합니다. 즉, i,j를 제외한 모든 변수들을 조건부로 통제한 후 i와 j 사이의 순수한 직접적 관계만을 측정하는 것인데요.


$$
\rho_{ij \mid V \setminus \{i, j\}} = 0
$$


다른 모든 변수들의 영향을 다 뺀 후, i와 j가 얼마나 직접적으로 연관되어 있는가?를 의미합니다. 즉, i와 j가 직접적으로 독립인지, 직접적으로 연결이 존재하는지만을 남깁니다. 만약 0이 아니라면, i와 j 사이에 직접 연결(엣지)가 있다고 알 수 있습니다.


$$
X = (X_1, \ldots, X_N)^T
$$


X는 전체 변수(노드)를 하나의 벡터로 표현한 것인데, 이 벡터가 다변량 정규분포를 따른다고 할 때, i와 j를 제외한 모든 변수를 조건으로 잡고 계산한 partial correlation이 0이라는 것은 i와 j가 나머지의 모든 변수를 이미 알고 있는 상태에서는 더 이상 아무런 직접적인 연관이 없다는 것을 의미합니다. 즉, **조건부 독립**임을 뜻합니다.



즉, GGM(가우시안 그래프 모델)은 partial correlation이 0이 아닌지로 정확하게 해석이 가능합니다. 이는 다변량 정규분포에서만 맞게 되는 성질이 됩니다.



모든 나머지 변수를 컨디셔닝한 partial correlation이 0이 아니면, i와 j 사이에 직접 연결을 둡니다. 이렇게 정의한 그래프를 conditional independence graph라고 합니다.

이 모델은 다변량 정규분포 + 조건부 독립 그래프 = Gaussian Graphical Model 이 됩니다.


$$
\text{The graph } G=(V,E) \text{ defined by: }\\
E = { {i, j} \in V^{(2)} : \rho_{ij|V\setminus{i,j}} \neq 0 }
$$




partial correlation(부분 상관계수)은 precision matrix(정밀도 행렬, Ω)의 원소로 간단하게 계산됩니다.


$$
\Omega=\Sigma^{-1} \\
\Sigma: \text{covariance matrix} \\
\Omega: \text{precision matrix, inverse covariance matrix}
$$


먼저, precision matrix(정밀도 행렬)이란, 공분산행렬의 역행렬을 의미합니다.



GGM에서 partial correlation을 계산하는 방법은

모든 나머지 변수를 조건으로 잡았을 때, i와 j의 부분 상관계수는 다음과 같이 표현됩니다.


$$
\rho_{ij \mid V \setminus \{i, j\}} = -\frac{ \omega_{ij} }{ \sqrt{ \omega_{ii} \omega_{jj} } }
$$


ω_{ij} : 정밀도 행렬의 (i,j) 원소입니다.



정밀도 행렬에서 (i, j) 원소가 0이 아니면, i와 j 사이에 직접 연결(엣지)가 있다고 판단합니다. 0이면, 조건부 독립이 성립합니다.



##### Frame Title

전통적인 방법은 공분산행렬과 정밀도행렬을 최대우도추정(likelihood-based) 등으로 계산합니다. 일반적으로 샘플 수가 변수의 개수보다 많다면 문제 없이 가능합니다. 그러나, 실제 데이터에서 변수 개수가 샘플 수보다 훨씬 많을 때에는 공분산행렬이 역행렬이 불가능하거나 불안정해져서 전통적인 방법이 아예 안통하거나 부정확하게 됩니다. 이런 상황에서는 Penalized Regression Methods를 사용하는 것이 일반적입니다. 가장 대표적인 방법은 LASSO 방법입니다.



Gaussian Model에서 0-평균 가정을 한다면, 아래와 같은 조건부 기대 수식을 나타낼 수 있습니다.


$$
\mathbb{E}[X_i \mid X_{-i}] = (\beta^{(-i)})^T X_{-i} \\
X_i : \text{response variable} \\
X_{-i}: N_v-1\text{ , covariate} \\
$$


GGM에서는 각 변수는 나머지 변수의 선형 조합으로 예측됩니다. 여기서 0-평균 가정이라 상수항은 없습니다.



회귀계수를 precision matrix와 연결 시킬 수 있습니다.


$$
\beta_j^{(-i)}=-\frac{\omega_{ij}}{\omega_{ii}}
$$


회귀 계수는 정밀도 행렬 원소(precision matrix)와 1:1로 연결되어 있습니다. 그래서 GGM에서는 그래프 추정 문제와 회귀계수 추정 문제가 거의 동등하게 다뤄집니다.



GGM에서 LASSO 회귀의 의미와 네트워크 추론에서의 해석을 알아보겠습니다.


$$
\hat{\beta}^{(-i)} = \underset{ \beta : \beta_i = 0 }{ \arg\min } 
\left\{
\sum_{k=1}^{n} (x_{ik} - \beta^T x_{-i, k})^2
+ \lambda \sum_{j \neq i} |\beta_j|
\right\}
$$


이 수식의 첫번째 항은 평균제곱오차(OLS, 잔차의 제곱)이고, 두번째 항은 LASSO penalty (L1-norm : 계수의 절댓값의 합)입니다. 람다는 penalty의 강도를 조절하고, 자기 자신에 대한 계수는 0으로 고정합니다.

만약, 람다가 0이면 (즉, penalty가 없으면) OLS 회귀와 같아서 대부분의 회귀 계수가 0이 아니게 됩니다. 모든 edge가 살아남아서 fully connected graph가 됩니다. penalty가 충분히 크면, 많은 회귀 계수들이 정확히 0이 되고, edge가 사라집니다. 희소 네트워크가 됩니다. 즉, LASSO penalty의 목적은 불필요한 edge(즉, 직접적 연관이 약한 것)는 0으로, 중요한 것만 네트워크에 남기는 역할을 합니다. 실제로 penalty를 너무 약하게 주면 edge가 너무 많고, 너무 세게 준다면 중요한 연결도 다 없어져서 적절한 balance가 중요합니다.

만약, penalty를 L1-norm에서 L2-norm 형태로 바뀐다면, 여전히 같은 형태가 되지 않습니다. 희소성이 떨어지고 모든 edge가 조금씩 남는 경향이 생깁니다. LASSO penalty는 L1 norm이기 때문에 "0으로 만드는 힘"이 강합니다.



즉, 베타가 0이 아니라면, (i, j) 사이에 edge가 존재하게 됩니다. 만약, 베타 회귀 계수가 0이면 X_j가 X_i를 설명하는데 아무런 역할도 하지 않게 됩니다. (i, j) 사이에 직접적인 edge가 없습니다. 

```R
> library(huge)
> set.seed(42)
> huge.out <- huge(Ecoli.expr)
Conducting Meinshausen & Buhlmann graph estimation (mb)....done
> # Use RIC (may under-select edges)
> huge.opt <- huge.select(huge.out, criterion = "ric")
Conducting rotation information criterion (ric) selection....done
Computing the optimal graph....done
> summary(huge.opt$refit)
   Length     Class      Mode 
    23409 ddiMatrix        S4 
> sum(huge.opt$refit != 0)
[1] 1246
```



이 R코드는 huge 패키지를 이용하여 LASSO 추정과 **RIC에 의한 최적 네트워크 선택 결과**를 요약해주고 있습니다.

23409는 최적 네트워크의 adjacency matrix의 전체 원소 개수이고, 1246개의 edge를 가지고 있습니다. 이는 양방향 네트워크여서 623개의 실제 고유 엣지 개수라고 할 수 있습니다.

이 네트워크(huge.opt$refit)는 Ecoli.expr 데이터에서 RIC 기준으로 λ를 고른 결과 희소성(엣지 수)이 어느 정도 조절된, 해석 가능한 **sparse graph 구조**입니다.





```R
> # Use stability selection (may over-select edges)
> huge.opt <- huge.select(huge.out, criterion = "stars")
Conducting Subsampling....done.                  
> g.huge <- graph_from_adjacency_matrix(huge.opt$refit, mode = "undirected")
> summary(g.huge)
IGRAPH fbc56f7 U--- 153 623 -- 
> # Compare with partial correlation network
> str(intersection(g.pcorr, g.huge))
Class 'igraph'  hidden list of 10
 $ : num 153
 $ : logi FALSE
 $ : num [1:25] 145 145 124 112 137 134 110 118 106 103 ...
 $ : num [1:25] 144 143 111 111 108 107 96 95 91 86 ...
 $ : NULL
 $ : NULL
 $ : NULL
 $ : NULL
 $ :List of 4
  ..$ : num [1:3] 1 0 1
  ..$ : list()
  ..$ : list()
  ..$ : Named list()
 $ :<environment: 0x00000139ca656920> 
> # Compare with known biological network
> str(intersection(g.regDB, g.huge, byname = FALSE))
Class 'igraph'  hidden list of 10
 $ : num 153
 $ : logi FALSE
 $ : num [1:19] 145 128 117 132 112 107 86 64 126 93 ...
 $ : num [1:19] 144 127 116 115 111 94 85 63 59 45 ...
 $ : NULL
 $ : NULL
 $ : NULL
 $ : NULL
 $ :List of 4
  ..$ : num [1:3] 1 0 1
  ..$ : list()
  ..$ :List of 1
  .. ..$ name: chr [1:153] "acrR_b0464_at" "ada_b2213_at" "adiY_b4116_at" "alpA_b2624_at" ...
  ..$ : Named list()
 $ :<environment: 0x00000139ca236ed8> 
```



여기서는 stability selection을 통해 구한 R코드입니다. 

partial correlation network와 huge(StARS) 네트워크의 교집합이 25개의 edge로 나타나고 있습니다. 즉, partial correlation 네트워크(25개 edge)와 StARS huge 네트워크(623개 edge)가 **공유하는 엣지 개수 = 25**입니다. **biological network(regDB)와 huge(StARS) 네트워크의 교집합**이 19개의 edge로 나타나고 있습니다. 즉, 실제 biological 네트워크와 StARS huge 네트워크가 **공유하는 엣지 개수 = 19**입니다.

실증분석에서는 데이터에 따라 다른 방식으로 공유하는 edge의 개수를 count합니다.



