---
published: true
layout: single
title:  "Development Economics : Basic Econometrics"
categories: Development
tag: Study
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 경제학부 개발경제학 수업 중 '**Development Economics : Basic Econometrics**'에 대한 내용을 담고 있습니다.

---

## Development Economics : Basic Econometrics

#### Introduction

---

##### Economic Questions

경제학에서의 질문은 대부분 두 개 이상의 변수(X와 Y)가 포함됩니다.

예를 들면,

1년 더 공부하면 생산성(수입)이 향상되는가?

경찰 수를 늘리면 범죄율에 어떤 영향이 있는가?

UOS 진로 개발 센터를 다니면 학생의 취업률에 어떤 영향이 있는가?

과 같은 질문을 던집니다.



##### The Regression Line

회귀 방정식은 x(explanatory variable, 독립변수)와 y(dependent variable, 종속변수) 간의 관계를 포함합니다.

![regression line](C:\Users\김충남\Desktop\2025-1\kse128-github-blog\KSE128.github.io\images\2025-03-27-develop_econometrics\regression line.PNG)
$$
\hat{y}=\beta_0+\beta_1x \\
\beta_0 : \text{intercept of y} \\
\beta_1 : \text{slope}
$$


##### Some notations and terminology

$$
log(wage)=\alpha+\beta(\text{Years of schooling})
$$

이 식은 교육 연수가 임금에 어떤 영향을 미치는지에 대한 회귀식입니다. 
$$
\beta\ =\ \frac{\Delta\ log(Wage)}{\Delta\ Years\ of \ Schooling}
$$
회귀식의 기울기는 교육 연수가 1년 증가할 때, 로그 임금의 변화량을 나타냅니다. 로그 변환된 임금 변수에 관련하여 기울기는 교육 연수가 1년 증가할 때 임금의 비율적 증가율을 의미합니다.

<img src="C:\Users\김충남\Desktop\2025-1\kse128-github-blog\KSE128.github.io\images\2025-03-27-develop_econometrics\graph_regression.PNG" alt="graph_regression" style="zoom: 40%;" />

이를 일반화하면, 임금 증가율은
$$
\% \Delta \text{Wage} \approx 100 \times \beta
$$
로 나타낼 수 있으며, 𝛽값에 100을 곱하면 교육 연수 증가에 따른 임금 증가율을 퍼센트 단위로 해석할 수 있습니다.  𝛼와 𝛽 모두 모집단의 미지의 모수로 그 값을 알지 못합니다. 따라서, 표본 데이터를 통해 추정합니다.

𝛼(절편, intercept)는 교육 연수가 0일 때의 로그 임금을 의미합니다. 
$$
\text{when we know} \ \ (x_1,y_1) \ \ \text{and} \ \ (x_2,y_2), \\ \\
\beta = \frac{y_2-y_1}{x_2-x_1}
$$
만약, 회귀선의 두 점에 대한 정보를 알고 있다면 𝛽(기울기, slope)값을 구할 수 있습니다.


$$
log(wage)=\alpha+\beta s
$$


회귀식에서, 𝛽값이 0.14라면, 교육연수가 1년 증가할 때, 임금이 약 14% 증가합니다. 교육 연수가 1년 더 많으면 임금이 평균적으로 약 14% 더 높을 것으로 예상할 수 있습니다.

𝛼값이 0.87이라면, 교육 연수가 하나도 없을 때, 로그 임금은 0.87으로 시간당 임금은 2.39입니다. 그러나, 현실적으로 교육 연수가 0인 경우는 거의 없기 때문에 절편을 해석하는 것은 의미있는 해석이라고 하지 않습니다.



##### The population Linear Regression Model

$$
Y_i=\alpha+\beta X_i+\epsilon_i, \ \ \text{where} \ \ i=1,2,\cdots,n \\
X : \text{independent variable, explanatory variable, regressor} \\
Y : \text{dependent variable, outcome variable} \\
\alpha : \text{intercept} \\
\beta : \text{slope} \\
\epsilon : \text{regression error}
$$

회귀분석에서 **ε(오차항, regression error)**은 **모델에서 설명되지 않은 부분**을 나타냅니다.  

<img src="C:\Users\김충남\Desktop\2025-1\kse128-github-blog\KSE128.github.io\images\2025-03-27-develop_econometrics\linear regression model.PNG" alt="linear regression model" style="zoom:50%;" />

최적의 회귀선은 주어진 데이터에 대해 가장 적절한 예측을 제공하는 선을 의미합니다. 위의 그래프에서는 회귀선은 데이터 점들과 가장 가까운 위치에 놓여 있으며, 데이터의 패턴을 가장 잘 설명하는 방향으로 설정되었습니다. 그래프에 표시되어있는 오차들은 개별 데이터 포인트들이 회귀선에서 얼마나 떨어져있는지를 나타내고 있습니다.



##### The Best Way to Fit a Line

위에서 언급한 최적의 회귀선 선택 방법은 **최소제곱법(OLS, Ordinary Least Squares)**로 을 **최소화하는 직선**을 찾는 것입니다. 

오차는 관측된 값과 회귀식으로 예측된 값 사이의 차이로 나타냅니다.
$$
\epsilon_i=Y_i-\hat{Y_i}=Y_i-(\alpha+\beta X_i)
$$


오차 제곱합을 구하면,
$$
SSE=\sum_{i=1}^{n}(Y_i-(\alpha+\beta X_i))^2
$$
형태로 나타납니다. 이 형태에서 𝛼와 𝛽를 결정하게 됩니다.

**OLS의 목표**는 𝛼와 𝛽를 선택하여 **SSE를 최소화**하는 것입니다.
$$
\min_{\alpha, \beta} \sum_{i=1}^{n} (Y_i - (\alpha + \beta X_i))^2
$$
이 과정에서 **𝛼와 𝛽의 최적 값**을 구할 수 있습니다.
$$
\hat{\beta} = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2} = \frac{S_{XY}}{S_X^2} \\
\hat{\alpha} = \bar{Y} - \hat{\beta} \bar{X}
$$
여기서, **X bar와 Y bar**는 각각 **X와 Y의 평균**을 나타냅니다. 

**hat**는 모집단의 파라미터를 추정한 값임을 구별하기 위해 사용됩니다.



**The Variance of OLS Estimator**

기울기와 절편에 대한 OLS 추정치의 분산은 다음과 같습니다.
$$
\sigma_{\hat{\beta}}^2=\frac{1}{n}\frac{var[(X_i-\mu_X)\epsilon_i]}{[var(X_i)]^2}
$$

$$
\hat{\sigma}^2_{\alpha} = \frac{1}{n} \cdot \frac{\text{var}[H_i \epsilon_i]}{[E(H_{i}^2)]^2}

\ \ where, \ 
H_i = 1 - (\frac{\mu_X}{E(X_i)^2}) X_i
$$

선형 회귀 분석의 경우 표준 오차는 더 복잡합니다. 그럼에도 불구하고, 표준오차는 여전히 표본크기와 반비례 관계에 있습니다. 표준 오차를 사용하면 가설 검정을 수행할 수 있습니다.



**Standard Errors**
$$
se(\hat{\beta}) = \frac{\hat{\sigma}}{\sqrt{\sum (x_i - \bar{x})^2}}, \quad \hat{\sigma}^2 = \frac{1}{n - 2} \sum \hat{\mu}^2
$$
표준 오차는 회귀 계수를 얼마나 정확하게 추정할 수 있는지에 대해 제공합니다. 표준오차가 작을 수록 계수 추정치가 더 정확합니다. 이때, 계수가 "통계적으로 유의미하다"고 말합니다. 이는 계수의 실제 값이 0과 다를 가능성이 매우 높다는 것을 의미합니다.





**Hypothesis Testing**

먼저 t-statistic











