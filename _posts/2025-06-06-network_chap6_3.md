---
published: true
layout: single
title:  "Network : Latent Network Models"
categories: Network
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Latent Network Models'**에 대한 내용을 담고 있습니다.

---

### Chapter 6 Statistical Models for Network Graphs (Part 3)



#### Latent Network Models

##### Introduction

우리는 앞에서 SBM(Stochastic Block Model)은 노드가 어떤 그룹에 속해있고, 그 그룹 간의 연결 확률로 네트워크를 설명했습니다. 이때, 그 그룹의 소속 정보 자체도 관측되지 않아 잠재변수로 추정했습니다. 이를 Stochastic Block Model이라고 합니다. 이 메커니즘을 더 확장한 모델이 Latent Network Models 입니다.

Latent Network Models의 핵심 아이디어는 관측되지 않는 특성(latent feature)이 노드 간 엣지 형성을 결정 짓는다고 가정합니다. SBM처럼 각 노드가 잠재 특성을 가지고 있고, 그 특성에 따라 연결될 확률이 달라집니다. SBM에서는 latent variable이 group membership (Z)이었는데, 이제는 그 latent feature가 더 유연해졌습니다. 예를 들어, 사회적 유치, 관심사 좌표, latent traits 등이 있습니다.  

또, 앞에서는 엣지의 생성 여부를 관측된 속성(성별, 나이)이나 완전한 랜덤성만 가지고 설명하는데, Latent Network Models은 각 노드가 보이지 않는 어떤 특징을 가지고 있고, 그 특성이 엣지가 생기는 데 영향을 준다고 가정합니다. Latent structure가 엣지 형성의 규칙성을 설명해주는 역할을 해줍니다. 마지막으로, 관측된 정보가 없더라도 잠재 공간(latent space) 모델은 네트워크 속 패턴이나 규칙성을 잘 포착할 수 있습니다.



##### Exchangeability

네트워크 모델에 latent variable을 도입하는건 exchangeability(교환 가능성) 라는 개념에 기반합니다. exchangeability는 노드를 재정렬해도 전체 그래프의 확률 분포가 바뀌지 않는 것을 뜻하며, 노드의 순서에는 의미가 없고, 오직 노드 간 관계(엣지)만 의미 있다는 전제로 나타납니다. 

만약, 그래프의 노드들이 exchangeable하다면, (노드의 번호나 순서를 바꿔도 본질적으로 같은 그래프로 본다!)

= 노드의 순서를 어떻게 바꾸더라도, 그래프 전체에 대한 확률 분포는 바뀌지 않는다!

이는, vertex의 순서를 바꿔놔도 n개의 vertex의 joint distribution에 영향을 끼치지 않습니다. 

위의 조건이 성립한다면, 각 엣지들은 아래와 같은 특정한 형태로 표현될 수 있습니다.


$$
Y_{ij}=h(\mu,u_i,u_j,\varepsilon_{ij}) \\
\mu : \text{global parameter} \\
u_i : \text{iid. latent variable} \\
\varepsilon_i : \text{iid pair specific noise terms} \\
h:\text{symmetric function in its second and third arguments}
$$


순서를 바꿔도 네트워크 확률 분포가 같다면 (exchangeability) , 전체 인접 행렬 Y는 노드별 latent variable과 엣지별 노이즈를 조합한 함수로 표현하게 됩니다.


$$
Y : h(\mu,u_i,u_j,\varepsilon_{ij})=-h(\mu,u_j,u_i,\varepsilon_{ij})
$$


h 함수의 경우 대칭함수로 이루어져있는데, 이를 거리 기반의 함수로 작성한다면, 함수 내부의 마이너스 부호를 반영하여 대칭 구조 안에서 방향성을 표현할 수 있게 됩니다.



이번엔 아래의 수식에 대한 구체적인 가정을 제시해보겠습니다!


$$
Y_{ij}=h(\mu,u_i,u_j,\varepsilon_{ij})
$$


첫번째는 엣지마다 주어지는 randomness(noise)는 표준 정규분포를 따르는 확률 변수라고 가정합니다.


$$
\varepsilon_{ij} \sim N(0,1)
$$


두번째는 노드 i와 j의 latent feature를 조합해서 얻는 함수는 대칭적이어야합니다. 


$$
\alpha(u_i,u_j)=\alpha(u_j,u_i)
$$


세번째는 h는 indicator 함수로 결과가 0보다 크다면 연결(1), 작다면 연결하지 않음(0)으로 나타냅니다. 즉,
$$
Y_{ij}
$$
는 특정 함수의 값이 0을 넘는지의 여부로 결정됩니다.

마지막으로, 파라미터 
$$
\mu
$$
는 관측된 공변량 
$$
x_{ij}
$$
의 선형 결합 
$$
x_{ij}^T\beta
$$
에 의해 확장됩니다. latent model 안에 관측된 공변량을 반영하여 연결 확률을 더 풍부하게 설명하는 모델입니다.



##### Latent Network Probit Model

4가지 가정 하에서 Latent Network Model의 수식 구조를 Probit 모델의 형태로 정리할 수 있습니다.

$$
P(Y_{ij} = 1 \mid X_{ij} = x_{ij}) = \Phi\left( \mu + x_{ij}^T \beta + \alpha(u_i, u_j) \right)
$$


즉, **관측변수(x_ij)**와 **숨겨진 특성(u_i, u_j = latent variable)**이 함께 확률을 결정하여 그 확률은 표준 정규 누적분포(CDF)를 통해 mapping돼서 엣지가 생길지 말지 결정합니다.

그렇다면, 어떤 요소가 1이 될 확률을 높이는지를 확인해봅시다. 

<img src="{{site.url}}\images\2025-06-06-network_chap6_3\probit.jpg" alt="probit" style="zoom:40%;" />

먼저, 노란 형광펜의 박스를 확인해보면 전체 probit 함수 내부, 즉
$$
\mu + x_{ij}^T \beta + \alpha(u_i, u_j)
$$
부분이 클수록 누적분포함수(CDF)의 출력값이 1에 가까워지면서, 연결될 확률이 높아진다는 의미를 담고 있습니다.

보라색 형광펜 부분을 보면, 
$$
x_{ij}^T\beta
$$
은 관측된 특성의 선형 효과로 x와 베타가 동일한 방향이면 그 항이 커지고 전체 확률도 커지게 됩니다.

마지막으로, 
$$
x_{ij}^T \beta + \alpha(u_i, u_j)
$$
부분은 전체 함수가 증가함수에 들어가서, 내부 합이 커질수록 엣지가 생길 확률이 증가합니다.

<img src="{{site.url}}\images\2025-06-06-network_chap6_3\probup.PNG" alt="probup" style="zoom: 67%;" />





앞의 Latent Network Probit Model을 기반으로 전체 네트워크의 likelihood(가능도)를 수식으로 표현하면, 다음과 같습니다.


$$
P(Y = y \mid X, u_1, \ldots, u_N) = \prod_{i < j} p_{ij}^{y_{ij}} (1 - p_{ij})^{1 - y_{ij}} \\
p_{ij}=P(Y_{ij}=1 \mid X_{ij}) \\
y_{ij} \in \{0,1\}
$$


여기서 p는 Probit 모델에서 예측된 확률이고, y는 실제 관측된 네트워크의 엣지 유무를 의미합니다. 

이 식에서 각 엣지는 독립적인 베르누이 분포를 따른다고 가정합니다. 전체 네트워크는 개별 Y들로 구성되므로, 전체 likelihood는 모든 i < j에 대해 확률을 곱한 형태로 나타납니다.



##### Specific Models of Latent Network


$$
P(Y_{ij} = 1 \mid X_{ij} = x_{ij}) = \Phi\left( \mu + x_{ij}^T \beta + \alpha(u_i, u_j) \right)
$$

$$
\alpha(u_i, u_j)
$$
의 구조에 따라 특정한 모델의 형태로 나타내어집니다.

먼저, SBM 기반의 Latent Class Model이 있습니다.


$$
u_i \in \{1, 2, \ldots, Q\}, \quad \alpha(u_i, u_j) = m_{u_i u_j}
$$


각 노드는 Q개의 클래스 중 하나에 속합니다. 클래스 간의 연결 경향은 Matrix M에 의해 결정됩니다. 이를 Stochastic Block Model과 같은 형태입니다.



두번째로, Latent Distance Model이 있습니다.


$$
u_i \in \mathbb{R}^Q, \quad \alpha(u_i, u_j) = -\|u_i - u_j\|
$$


각 노드는 Q차원 공간상의 위치 벡터 (u_i)를 가집니다. u_i와 u_j가 가까울수록 연결 확률이 높습니다. 거리가 가까워지면 알파값이 커집니다. 이는 사회적 거리 또는 homophily를 잘 설명하는 모델이라고 할 수 있습니다.



마지막으로 Eigenmodel 입니다.


$$
u_i \in \mathbb{R}^Q, \quad \alpha(u_i, u_j) = u_i^T \Lambda u_j \quad (\Lambda\ \text{is diagonal})
$$


각 노드는 여전히 Q차원의 벡터 u_i인데, 거리 기반이 아니라 내적 구조입니다.

이 모델은 구조적 동등성도 설명이 가능합니다. 즉, 같은 방식으로 다른 노드들과 연결되는지를 설명합니다.

Eigenmodel은 위 두 모델을 일반화합니다. eigenmodel은 더 유연하고 포괄적인 latent interaction 구조를 표현할 수 있습니다.



##### Application of Latent Network Models

```R
> A <- get.adjacency(lazega, sparse=FALSE)
> # Model with no covariates, Q = 2
> lazega.leig.fit1 <- eigenmodel_mcmc(A, R=2, S=11000, burn=10000)

> # Add common practice covariate : 같은 practice 여부
> same.prac <- with(v.attr.lazega, Practice %o% Practice)
> same.prac <- matrix(as.numeric(same.prac %in% c(1,4,9)), 36,36)
> same.prac <- array(same.prac, dim=c(36,36,1))
> # MCMC 적합
> lazega.leig.fit2 <- eigenmodel_mcmc(A, same.prac, R=2, S=11000, burn=10000)

> # Add office location covariate : 같은 office 여부
> same.off <- with(v.attr.lazega, Office %o% Office)
> same.off <- matrix(as.numeric(same.off %in% c(1,4,9)), 36,36)
> same.off <- array(same.off, dim=c(36,36,1))
> lazega.leig.fit3 <- eigenmodel_mcmc(A, same.off, R=2, S=11000, burn=10000)

> # Extract eigenvectors
> lat.sp.1 <- eigen(lazega.leig.fit1$ULU_postmean)$vec[, 1:2]
> lat.sp.2 <- eigen(lazega.leig.fit2$ULU_postmean)$vec[, 1:2]
> lat.sp.3 <- eigen(lazega.leig.fit3$ULU_postmean)$vec[, 1:2]

> # Plot
> colbar <- c("red", "dodgerblue", "goldenrod")
> v.colors <- colbar[V(lazega)$Office]
> v.shapes <- c("circle", "square")[V(lazega)$Practice]
> v.size <- 3.5 * sqrt(V(lazega)$Years)
> v.label <- V(lazega)$Seniority
> plot(lazega, layout=lat.sp.1, vertex.color=v.colors,
+      vertex.shape=v.shapes, vertex.size=v.size,
+      vertex.label=v.label)

> # 각 모델의 Lambda 평균 확인
> apply(lazega.leig.fit1$L_postsamp, 2, mean)
[1] 0.7582598 0.5253324
> apply(lazega.leig.fit2$L_postsamp, 2, mean)
[1]  0.8706554 -0.2211593
> apply(lazega.leig.fit3$L_postsamp, 2, mean)
[1] -0.1240265  0.3637222

> perm.index <- sample(1:630)
> nfolds <- 5; nmiss <- 630 / nfolds
> Avec <- A[lower.tri(A)]; Avec.pred1 <- numeric(length(Avec))
> for(i in 1:nfolds){
+   miss.index <- seq((i-1)*nmiss+1, i*nmiss)
+   A.miss.index <- perm.index[miss.index]
+   Avec.temp <- Avec
+   Avec.temp[A.miss.index] <- NA
+   Atemp <- matrix(0,36,36)
+   Atemp[lower.tri(Atemp)] <- Avec.temp
+   Atemp <- Atemp + t(Atemp)
+   model1.fit <- eigenmodel_mcmc(Atemp, R=2, S=11000, burn=10000)
+   model1.pred <- model1.fit$Y_postmean
+   Avec.pred1[A.miss.index] <- model1.pred[lower.tri(model1.pred)][A.miss.index]
+ }

> # ROC 평가
> library(ROCR)
> pred1 <- prediction(Avec.pred1, Avec)
> perf1 <- performance(pred1, "tpr", "fpr")
> plot(perf1, col="blue", lwd=3)

# AUC 출력
> performance(pred1, "auc")@y.values
[1] 0.820515

```



첫번째는 같은 practice 범주에 속한 노드쌍을 1로 표시한 공변량을 추가한 모델입니다. 두번째는 같은 office에 속한 노드쌍을 기반으로 공변량을 구성한 모델입니다. 각 모델을 적합한 후 eigen decomposition하여 상위 2개 eigenvector를 추출하고 이를 latent space 좌표로 사용합니다. 네트워크는 해당 latent space 상에 시각화되며, 노드의 색상은 사무실(Office), 모양은 법률 practice, 크기는 근속연수(Years), 라벨은 선임도(Seniority)를 나타냅니다. ROCR 패키지를 활용해 예측 결과의 ROC 곡선을 시각화하고, AUC (Area Under the Curve) 값을 계산합니다. 이 분석에서는 AUC = 0.82로 나타나, Eigenmodel이 네트워크 구조를 상당히 높은 정확도로 예측하고 있음을 보여줍니다.

![eigenplot]({{site.url}}\images\2025-06-06-network_chap6_3\eigenplot.PNG)

<img src="{{site.url}}\images\2025-06-06-network_chap6_3\ROC_curve.PNG" alt="ROC_curve" style="zoom: 67%;" />





이 그래프는 ROC(Receiver Operating Characteristic) 곡선이고, lazega 네트워크에 대해 적합한 세 가지 eigenmodel의 **예측 성능(goodness-of-fit)**을 비교한 그림입니다. 이 곡선은 모델이 얼마나 정확하게 연결을 예측하는지를 보여줍니다.

각각 다음 세 가지 설정을 사용합니다:
 (i) **쌍별(pair-specific) 공변량을 포함하지 않은 모델** (파란색),
 (ii) **같은 법률 practice를 공유하는지 여부를 공변량으로 포함한 모델** (빨간색),
 (iii) **같은 사무실 위치를 공유하는지 여부를 공변량으로 포함한 모델** (노란색).



세 곡선 모두 위로 볼록하며, 좌상단에 가까운 형태를 나타납니다. 이는 예측력이 상당히 좋음을 나타냅니다. **AUC**는 0.8에 가까워 엣지 예측에서 일관되게 높은 성능을 보입니다. ROC 곡선을 통해 latent structure만으로도 상당히 많은 구조적 정보를 설명할 수 있음을 보입니다. 공변량을 추가해도 예측 성능은 유사하게 유지되는 것을 ROC 곡선을 통해 확인할 수 있었습니다.  

가장 이상적인 경우는 그래프 좌상단에 그린 꺾인 선으로 완벽한 모델의 ROC 곡선입니다. (TPR = 1, FPR = 0)

또한, 그래프 아래 면적이 클수록 더 좋은 결과입니다. 

여기서 확인할 것은 TPR(진짜 positive 맞추기)와 FPR(거짓 positive 줄이기)은 동시에 최적화하기 어렵습니다. 하나가 커지면 다른 하나도 커질 수 있기 때문에 균형점이 필요합니다.



confusion matrix는 모델의 **예측 결과를 네 가지 경우(TP, FP, TN, FN)**로 정리한 표로, ROC 곡선 계산의 출발점이며, 네트워크 예측에서 **성공/실패 케이스를 분류해주는 핵심 도구**입니다.

<img src="{{site.url}}\images\2025-06-06-network_chap6_3\matrix.PNG" alt="matrix" style="zoom:75%;" />

| 이름              | 의미       | 설명                      |
| ----------------- | ---------- | ------------------------- |
| TP(True Positive) | 예측 성공! | 실제 1이고 1로 예측함     |
| FP                | 거짓 경고  | 실제는 0인데 1로 예측함   |
| TN                | 예측 성공! | 실제 0이고 0으로 예측함   |
| FN                | 놓침       | 실제는 1인데 0으로 예측함 |



