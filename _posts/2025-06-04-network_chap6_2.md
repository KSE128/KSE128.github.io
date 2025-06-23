---
published: true
layout: single
title:  "Network : Network Block Models"
categories: Network
toc: true
use_math: true
---



이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Network Block Models' **에 대한 내용을 담고 있습니다.

---

### Chapter 6 Statistical Models for Network Graphs (Part 2)



#### Network Block Models

앞에서 ERGM의 구조는 전통적인 회귀 모델과 유사하다는 점을 확인했습니다. 이번엔 Classical Mixture Models과 유사한 Network Block Models에 대해 알아보겠습니다.



##### Introduction

**Classical Mixture Models**

Classical Mixture Model(고전적 혼합 모형)은 여러 개의 구성 확률 분포의 가중합으로 하나의 확률 분포를 표현하는 방법입니다. 즉, 어떤 데이터가 하나의 분포가 아니라 여러 개의 하위 집단에서 나왔다고 가정할 때 혼합 모델을 사용하면 좋습니다.

이 모델을 아래와 같은 수식으로 나타낼 수 있습니다.


$$
f(x)= \sum_{q=1}^Q \alpha_qf_q(x) , \\
f_q(x): \text{the component densities} \\
\alpha_q: \text{mixing proportions such that}\ \alpha_q \geq0 \ \ \text{and} \ \ \sum_q\alpha_q=1
$$


즉, 데이터가 여러 하위 그룹에서 나온 것으로 보일 때, 그 하위 그룹은 서로 다른 분포를 가지고 있고, 전체는 그들의 가중 평균으로 표현할 수 있습니다.

Classical Mixture Model은 데이터가 여러 하위 집단에서 나온 것으로 여겨지고 각 하위 집단이 고유한 분포로 설명될 때 유용합니다.



**Network Block Models**

각 노드 i는 전체 노드 집합 V 중 하나이며, Q개의 클래스(블록) 중 하나에 속한다고 가정합니다.


$$
C_1,C_2,\cdots,C_Q
$$


각 노드 i는 어떤 클래스에 속해 있는지 q(i)가 주어지게 됩니다. 

인접 행렬 Y의 각 원소는 노드 i와 j가 각각 클래스 q와 r에 속할 때, 연결 여부는 확률을 따르는 독립적인 베르누이 분포로 모델링됩니다.


$$
Y_{ij} \sim \mathrm{Bernoulli}(\pi_{qr})
$$


즉, 연결될 확률은 단지 노드들이 속한 클래스 쌍(q, r)에 따라 달라지게 됩니다.



만약, Undirected Graph의 경우,
$$
Y_{ij}=Y_{ji}
$$
이 연결 확률은 대칭이 됩니다. 이는, 
$$
\pi_{qr}=\pi_{rq}
$$
가 됩니다.





Network Block Model은 ERGM의 특수 형태로 표현할 수 있습니다.


$$
P_{\theta}(Y = y) = \frac{1}{\kappa} \exp\left\{ \sum_{q, r} \theta_{qr} L_{qr}(y) \right\} \\
\text{where} \ \ L_{qr}(y) \ \ \text{the number of edges in y connecting classes q and r.}
$$


L(y)는 y에서 블록 q와 r를 연결하는 엣지의 개수입니다.

이 모델은 네트워크 y가 생성될 확률을 블록 간 연결 수와 그것에 곱해지만 모수의 합으로 설명합니다. 이는 ERGM처럼 블록 간 연결 수를 통계량으로 설정해 exponential family로 표현 가능합니다.



##### Network Block Models : Stochastic Block Model (SBM)

앞에서 다룬 Network Block Models는 현실에서는 **블록의 소속 정보를 알기 어렵기 때문**에 많이 사용하지 않습니다. 

그래서 Stochastic Block Model을 사용합니다.

네트워크의 노드들이 어떤 그룹(블록)에 속해있다고 가정해봅시다. 그러나, 그 블록 소속 정보는 관측되지 않습니다. 대신, 확률적으로 어떤 블록에 속할지 추정하고 블록 간 연결 확률에 따라 네트워크가 형성된다고 보는 모델을 Stochastic Block Model이라고 합니다. 

Stochastic Block Model을 적용하기 위해서는 가정을 세워봅시다. 

먼저,
$$
Z_{iq}=1 \ \text{if vertex}\ i \ \text{is in class} \ q, \ 0 \ \text{otherwise}
$$


노드 i가 블록 q에 속하면 Z_{iq}=1이고 그렇지 않으면 0입니다.



두번째로,
$$
P(Z_{iq} = 1) = \alpha_q, \qquad \sum_q \alpha_q = 1
$$


각 노드는 독립적으로 블록 q에 속할 확률 
$$
\alpha_q
$$
를 가집니다. 즉, 블록 소속도 모수화된 확률 분포로 모델링굅니다.



마지막으로, 
$$
Y_{ij} \sim \mathrm{Bernoulli}(\pi_{qr}) \quad \text{if } Z_{iq} = 1,\, Z_{jr} = 1
$$


노드 i와 j의 블록이 각각 q와 r일 경우, 이 두 노드 사이의 연결 
$$
Y_{ij}
$$
는 확률
$$
\pi_{qr}
$$
로 발생합니다.



그리고 Stochastic Block Model에서 추정해야할 대상은 
$$
Z_i
$$
(노드 i가 어느 블록에 속하는지)와 
$$
\pi_{qr}
$$
(블록 q와 r 사이의 엣지 생성 확률) 입니다. 즉, 블럭의 소속도 모르고 연결 확률도 모르는 상태입니다. 

이 추정은 연결 정보 
$$
Y_{ij}
$$
만 보고 동시에 추정하는 모형입니다.



Stochastic Block Model (SBM)은 고전적인 혼합모형(Classical Mixture Model)과 개념적으로 유사합니다.

Classical Mixture Model에서는 데이터를 생성할 때, 먼저 어떤 숨겨진 클래스를 선택하고, 그 클래스에 맞는 확률분포로부터 데이터를 샘플링하는 방식입니다. SBM 모델도 이와 비슷한 속성을 가지고 있습니다.

각 노드는 어떤 숨겨진 블록에 속할 확률을 가지고 있습니다. 이는 mixture weights와 동일한 개념을 가집니다. 어떤 두 노드가 연결될지는 두 노드가 속한 클래스 쌍에 따라 달라집니다. 이는 클래스별 확률 분포에서 샘플링하는 것과 동일합니다. 관측된 네트워크는 여러 블록 간 연결 패턴의 혼합 결과이고, 마치 mixture model이 여러 분포의 조합으로 데이터를 설명하는 것과 같습니다.

이제는 네트워크 데이터에 대해 Stochastic Block Model(SBM)을 fit하는 과정을 알아보겠습니다.

우선 fblog 데이터를 기반으로 노드들을 블록(그룹)으로 나누고 블록 간 연결 패턴을 추정하기 위해 SBM을 학습합니다.

```R
> A.fblog <- as.matrix(as_adjacency_matrix(fblog, sparse = FALSE))
> fblog.sbm <- BM_bernoulli("SBM_sym", A.fblog, verbosity = 0)
> fblog.sbm$estimate()

> ICLs <- fblog.sbm$ICL
> Q <- which.max(ICLs)
> Q
[1] 10
```

<img src="{{site.url}}\images\2025-06-04-network_chap6_2\ICL.PNG" alt="ICL" style="zoom:50%;" />

위의 R코드를 통해 이 데이터에서는 블록 수 Q = 10일 때 모델이 가장 좋았다는 결과가 도출되었습니다. 이는 네트워크 속 숨겨진 집단 구조를 10개로 나누는 것이 가장 설득력 있는 설명이라는 것을 의미합니다.



```R
> Z <- fblog.sbm$memberships[[Q]]$Z
There were 15 warnings (use warnings() to see them)
> round(head(Z), 4)
       [,1]  [,2]  [,3]  [,4]  [,5]  [,6]  [,7]  [,8]  [,9] [,10]
[1,] 0.9953 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04
[2,] 0.9953 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04
[3,] 0.9953 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04
[4,] 0.9953 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04
[5,] 0.9953 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04
[6,] 0.9953 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04 5e-04
> 
> cl.labs <- apply(Z, 1, which.max)
> nv <- vcount(fblog)
> summary(Z[cbind(1:nv, cl.labs)])
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
 0.8586  0.9953  0.9953  0.9938  0.9953  0.9953 
> 
> cl.cnts <- as.vector(table(cl.labs))
> alpha <- cl.cnts / nv
> round(alpha, 4)
 [1] 0.1823 0.1406 0.0573 0.1094 0.1250 0.1302 0.0312 0.0365 0.0677 0.1198
> 
> Pi.mat <- fblog.sbm$model_parameters[[Q]]$pi
> round(Pi.mat[3,], 4)
 [1] 0.0030 0.0074 0.9102 0.0009 0.0009 0.0365 0.0178 0.0025 0.0432 0.0012
```



먼저, `Z[i,q]`는 노드 i가 블록 q에 속할 확률입니다. 각 노드의 첫번째 열은 0.9953이고 나머지는 0.0005로 거의 블록 1에 강하게 속합니다. 

`summary(Z[cbind(1:nv, cl.labs)])`는 클래스 소속 확률의 요약 통계량으로 대부분의 노드는 자신이 속한 블록에 대한 확률이 0.9953으로 매우 높습니다. 가장 낮은 경우도 0.8586으로 매우 강한 소속감을 지닙니다. 이는 군집 추정이 확실하고 안정적임을 의미합니다.

또한, 다른 파라미터들도 우리의 관심사입니다.

`round(alpha, 4)`는 각 블록에 속한 노드의 수를 비율화 하여 계산한 것인데, 전체 노드 중 블록 1에는 18.2%, 블록 2에는 14.1% ... 로 블록 7-10는 작습니다. 이를 통해 6개의 큰 블록과 4개의 작은 블록, 즉 비균형적인 커뮤니티 구조가 반영됩니다.

Pi.mat행렬은 노드 i가 블록 q에, 노드 j가 블록 r에 속해있을 때, 두 노드가 연결될 확률을 의미합니다. 여기서의 dimension은 Q * Q 행렬입니다.

`> round(Pi.mat[3,], 4)
 [1] 0.0030 0.0074 0.9102 0.0009 0.0009 0.0365 0.0178 0.0025 0.0432 0.0012`

이 행렬은 블록 3과 다른 블록 간의 연결 확률을 나타내는데, 블록 3은 내부 연결이 매우 강(91%)하나 외부 블록과는 거의 연결되지 않습니다.

```R
> ntrials <- 1000
> Pi.mat <- (t(Pi.mat) + Pi.mat) / 2
> deg.summ <- list(ntrials)
> for(i in (1:ntrials)){
+   blk.sz <- rmultinom(1, nv, alpha)
+   g.sbm <- sample_sbm(nv, pref.matrix=Pi.mat, block.sizes=blk.sz,
+                       directed=FALSE)
+   deg.summ[[i]] <- summary(degree(g.sbm))
+   }
> 
> Reduce('+', deg.summ) / ntrials
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  1.905   9.159  13.188  15.211  19.030  49.690 
> summary(degree(fblog))
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   2.00    8.00   13.00   14.91   18.00   56.00 
```

이 코드는 Stochastic Block Model (SBM)이 실제 네트워크 데이터를 잘 설명하는지를 시뮬레이션과 요약 통계량 비교를 통해 확인하는 과정입니다. SBM으로 1000개의 가짜 네트워크를 생성하고, 각 시뮬레이션 네트워크의 노드 degree 요약 통계량을 계산한 후, 그것을 실제 네트워크의 degree 분포와 비교하는 것인데, 시뮬레이션 결과와 실제 값이 전반적으로 매우 유사합니다. 이는 SBM이 degree 분포를 잘 재현하고 있다는 의미를 담고 있습니다.

```R
> edges <- as_edgelist(fblog, names=FALSE)
> neworder <- order(cl.labs)
> m <- t(matrix(order(neworder)[as.numeric(edges)], 2))
> plot(1, 1, xlim = c(0, nv + 1), ylim = c(nv + 1, 0),
+      type = "n", axes = FALSE, xlab = "Classes",
+      ylab = "Classes",
+      main = "Reorganized Adjacency matrix")
> rect(m[,2]-0.5, m[,1]-0.5, m[,2]+0.5, m[,1]+0.5, col=1)
> rect(m[,1]-0.5, m[,2]-0.5, m[,1]+0.5, m[,2]+0.5, col=1)
> cl.lim <- cl.cnts
> cl.lim <- cumsum(cl.lim)[1:(length(cl.lim)-1)] + 0.5
> clip(0, nv+1, nv+1, 0)
> abline(v=c(0.5, cl.lim, nv+0.5),
+        h=c(0.5, cl.lim, nv+0.5), col="red")
```

<img src="{{site.url}}\images\2025-06-04-network_chap6_2\adjacency.PNG" alt="adjacency" style="zoom: 50%;" />

이 코드는 SBM으로 추정한 블록 구조에 따라 노드를 재정렬한 후, 인접 행렬의 블록 구조(커뮤니티 패턴)을 시각화합니다. 빨간 선은 군집의 경계이고 검은 칸은 엣지의 존재를 나타냅니다. 대각선의 블록이 진하고, 비대각선 블록이 연한 것으로 보아, 내부 연결이 강하나 블록 간 연결은 약하다는 것을 의미합니다.



```R
> g.cl <- graph_from_adjacency_matrix(Pi.mat,
+                                     mode="undirected",
+                                     weighted=TRUE)
> # Set necessary parameters
> vsize <- 100*sqrt(alpha)
> ewidth <- 10*E(g.cl)$weight
> PolP <- V(fblog)$PolParty
> class.by.PolP <- as.matrix(table(cl.labs, PolP))
> pie.vals <- lapply(1:Q, function(i)
+   as.vector(class.by.PolP[i,]))
> my.cols <- topo.colors(length(unique(PolP)))
> # Plot
> plot(g.cl, edge.width=ewidth,
+      vertex.shape="pie", vertex.pie=pie.vals,
+      vertex.pie.color=list(my.cols),
+      vertex.size=vsize, vertex.label.dist=0.1*vsize,
+      vertex.label.degree=pi)
> # Add a legend
> my.names <- names(table(PolP))
> my.names[2] <- "Comm. Anal."
> my.names[5] <- "PR de G"
> legend(x="topleft", my.names, fill=my.cols, bty="n")
```



<img src="{{site.url}}\images\2025-06-04-network_chap6_2\visualize.PNG" alt="visualize" style="zoom: 50%;" />

이 과정은 SBM에서 추정한 블록 간 연결 네트워크를 pie chart 형태의 노드로 시각화하는데, 각 노드는 정당 분포를 파이로 나타내고, 엣지는 블록 간 연결 확률에 따라 두께를 다르게 표현합니다. 블록(클러스터) 간 연결 구조를 그래프로 나타내면서, 각 블록의 내부 구성을 파이차트로 시각화합니다.
