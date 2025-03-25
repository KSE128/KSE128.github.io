---
published: true
layout: single
title:  "Panel Data(패널 데이터)"
categories: Economics
tag: Activity
toc: true
use_math: true
---

이 포스팅은 **Panel Data(패널 데이터)**에 대한 내용을 담고 있습니다.

---

### Panel Data(패널 데이터)



#### Introduction

**Panel Data(패널 데이터)**는 **동일한 개체**(예: 개인, 기업, 국가 등)에 대해 여러 시점에 걸쳐 수집된 데이터를 의미합니다. **Cross-sectional Data(단면 데이터)와 Time-series Data(시계열 데이터)**의 특징을 둘 다 가지고 있습니다. Panel Data의 주요 특징은 **같은 개체를 반복 관측할 수 있다는 점**입니다. 예를 들어, 100명의 근로자의 임금, 교육 수준, 근속 연수 등을 5년 동안 매년 조사한다면, 동일한 100명의 근로자를 지속적으로 추적하는 형태가 됩니다. 또한, **시간에 따른 변화를 고려할 수 있**습니다. 개체마다 **시간이 지나면서 변하는 특성**을 분석할 수 있으며, 개인의 임금이 교육 수준이나 근속 연수에 따라 어떻게 변화하는지 연구할 수 있습니다. 마지막으로, **개체별 고유한 특성**(불변 요소)을 **통제**할 수 있는 방법을 제공합니다. 성격과 능력과 같이 시간이 지나도 변하지 않는 요인이 존재할 수 있으며, 이는 패널 데이터를 활용하여 변하지 않는 요인의 영향을 제거하고 정확한 분석이 가능합니다.



#### Two-Period Panel Data

---

다중회귀분석에서는 종속 변수에 영향을 미치는 모든 요인을 통제하기 어려운 경우가 많습니다. 모든 요인들을 통제하기 위해서는 종속 변수의 시차 값을 포함하는 것입니다.

예를 들어, 범죄율(crime)을 분석할 때, 이전 시점의 범죄율을 회귀식에 포함하면, **누락된 변수 문제**를 **일부 해결**할 수 있어 설명되지 않는 요인의 영향을 부분적으로 완화할 수 있습니다.
$$
crime = \beta_0 + \ \beta_1unem \ + \ \beta_2expend + \ \beta_3crime_{-1}\ +\ u
$$


또 다른 접근법은 종속 변수에 영향을 미치는 요인을 **시간에 따라 변하지 않는 요인**과 **변하는 요인**으로 나누어 분석하는 것입니다. 이를 위해 패널 데이터를 활용한 **고정효과 모델(fixed effect model)**을 이용하게 됩니다. 패널 데이터는 두 개의 시점(t=1,2)에 대한 데이터를 사용합니다. 이 시점은 인접하지 않아도 되는데, t=1은 더 이른 시점이어야합니다.
$$
y_{it}=\beta_0+ \ \delta_0d2_t\ + \ \beta_1x_{it} + \ a_i \ + \ u_{it} \\
i : \text{cross-sectional unit} \\
t : \text{time period} \\
d2_t : \text{time dummy variable} \\
d2_t =0 \ \text{(t=1)} \\
d2_t =1 \ \text{(t=2)} \\
a_i : \text{fixed effect(unobserved effect)} \\
u_{it} : \text{idiosyncratic error}
$$
fixed effect(고정효과)의 경우, 시간에 따라 변하지는 않지만 **개별 단위에 고유한 특성**을 나타냅니다. 



예를 들어,
$$
wage_{it}=\beta_0+ \ \delta_0d2_t\ + \ \beta_1educ_{it} + \beta_2exper_{it} + \ a_i \ + \ u_{it}
$$
근로자의 임금 데이터를 분석할 때, 개인의 능력이나 성격 같은 요소는 **시간에 따라 변하지 않는 특성**이므로 fixed effect 부분에 포함될 수 있습니다. 고정효과의 경우 **unobserved heterogeneity, unobserved effect(관측되지 않은 이질성, 관측되지 않은 효과)**라고도 불립니다.

반면, u 부분의 경우 **time-varying error(시간에 따라 변하는 요인)**으로, 경제 상황이나 개인의 경험 증가 등이 해당될 수 있으며, 이를 **idiosyncratic error(이질적 오차)**라고 합니다. u 부분은 시간이 지남에 따라 변화하고 **y에 영향을 미치는 관찰되지 않는 요소**를 나타내고 있습니다.

고정효과 모델을 사용하면, 시간에 따라 변하지 않는 unobserved effect를 제거할 수 있어 정확한 인과관계를 추정할 수 있게 됩니다.



#### Two-Period Panel Data : Pooled OLS

---

Pooled OLS는 두 개의 시점에 대한 패널 데이터가 주어졌을 때, 두 기간의 데이터를 결합하여 OLS 회귀를 수행하는 방식입니다.
$$
y_{it}=\beta_0+ \ \delta_0d2_t\ + \ \beta_1x_{it} + v_{it}, \ \ t=1,2 \\
v_{it} = a_i \ + \ u_{it}  \ \ \ (\text{composite error}) \\
$$
즉, v는 개별 단위에 따라 고정된 unobserved effect(a)와 시점에 따라 변화하는 오차(u)로 구성됩니다.


$$
Cov(x_{it}, a_i)=0
$$
Pooled OLS를 사용하려면, **설명변수와 unobserved effect의 상관관계가 없다**는 가정을 만족해야합니다. 만약, 둘의 상관관계가 존재한다면, Pooled OLS의 추정치는 bias(편향)과 inconsistency(비일관성) 문제가 발생하며, 계수 추정값을 신뢰할 수 없게 됩니다. 이 편향을 **heterogeneity bias(이질성 편향)** 또는 **omitted variable bias(누락 변수 편향)**이라고 합니다. 이러한 편향이 발생할 경우, Cross-sectional Data를 사용했을 때 크게 다르지 않은 결과가 나타날 수도 있습니다.  



#### Two-Period Panel Data : First Differencing

---

Pooled OLS를 사용했을 때 개인별 고유한 효과(a)가 설명변수와 상관이 있더라도, 패널 데이터를 사용하는 주된 이유는 둘의 상관관계가 존재해도 이를 제거하고 정확한 인과관계를 추정하기 위함입니다.



같은 개체에 대해 두 시점에 대한 방정식을 작성하면,
$$
y_{i2} = (\beta_0 + \delta_0) + \beta_1 x_{i2} + a_i + u_{i2}
$$

$$
y_{i1} = \beta_0 + \beta_1 x_{i1} + a_i + u_{i1}
$$

두 식의 차이를 계산하면:

$$
y_{i2} - y_{i1} = (\beta_0 + \delta_0) - \beta_0 + \beta_1 (x_{i2} - x_{i1}) + (a_i - a_i) + (u_{i2} - u_{i1})
$$

즉,

$$
\Delta y_i = \delta_0 + \beta_1 \Delta x_i + \Delta u_i
$$
위의 식을 1차 차분 회귀식이라고 합니다. 1차 차분 회귀식을 확인하면, 개별 단위에 따라 고정된 unobserved effect (a)가 제거되었음을 확인할 수 있습니다. 설명변수가 unobserved effect(a)을 통제한 상태에서 결과 변수에 미치는 영향을 명확하게 보여줍니다.  

이때, 편향되지 않고 일치하는 추정량을 얻기 위해서는 몇 가지 가정이 필요합니다.



##### Strict Exogeneity

**강외생성 조건**을 만족해야 합니다.
$$
E(u_{it} \mid X_i, a_i) = 0 \quad \text{for all } i \text{ and } t
$$

여기서  
$$
X_i = \{ x_{itj} \} \\
j:\text{the number of independent variable}
$$


오차항이 설명변수와 개별 고정효과를 알고 있을 때 기대값이 0이라고 가정합니다.
$$
E(u_{it} \mid X_i, a_i) = 0
$$


위의 가정에서 양변을 X에 대해 다시 기대값을 취하면, 
$$
E(E(u_{it} \mid X_ia_i)\ |\ X_i) = E(u_{it} \ | \ X_i)=0
$$
따라서 고정효과를 제거한 후에도 설명변수에 대해 조건부 기대값이 0이 됨을 알 수 있습니다.



위의 결과로부터, 서로 다른 시점 t와 s에서의 오차항 간에는 공분산이 0이어야합니다. 

즉,
$$
Cov(u_{it},u_{is})=0
$$
이는 오차항이 시점 간 상관관계가 없고, 자기 상관(serial correlation)이 없다는 의미를 가지고 있습니다.

이 가정은 설명변수가 시간에 따라 변화하는 오차항(u)과 상관이 없음을 의미합니다.



비슷한 맥락에서 1차 차분 후에도 오차항과 설명변수가 독립적이어야합니다. 패널 데이터에서 1차 차분을 하면 개별 고정효과가 제거됩니다. 따라서 **차분된 오차항은 차분된 설명변수와 독립**적이어야 합니다. 

이를 수식으로 표현하면,
$$
E(\Delta u_i \mid \Delta X_i) = 0
$$

오차항의 1차 차분을 전개했을 때,
$$
E(u_{i2}-u_{i1} \ | \  x_{i2}-x_{i1})=0
$$
두 시점의 오차항 차이가 두 시점의 설명변수 차이에 대해 기대값이 0이어야 합니다.



이를 조건부 기댓값의 법칙을 이용하여 표현하면 다음과 같은 결과가 나타납니다.
$$
E[E(u_{i2}-u_{i1} \ | \ X_i) \ | \ x_{i2}-x_{i1}]=0
$$
먼저, 전체 설명변수 집합을 알고 있을 때, 오차항의 차분의 기대값을 계산하는 과정으로 고정효과가 제거된 후에도 기대값이 0이므로, 
$$
E(u_{i2}-u_{i1} \ | \ X_i)=0
$$
이 성립합니다.



첫번째 기대값이 이미 0이므로, 다시 조건부 기대값을 계산하면,
$$
E[\ 0\ | \ x_{i2}-x_{i1}]=0
$$
이 성립합니다.

따라서, 
$$
E(u_{i2}-u_{i1} \ | \  x_{i2}-x_{i1})=0
$$
가 성립합니다.





즉, 1차 차분 후에도 설명변수와 오차항이 상관이 없어야합니다. 이를 다시 표현하면,
$$
\text{Cov}(\Delta x_i, \Delta u_i) = 0
$$

이 성립해야 합니다.

설명변수는 고정효과와는 상관관계가 존재해도 되지만, **시점에 따라 변화하는 오차와는 상관관계를 가지면 안됩**니다. 특히, 설명변수에 종속변수의 시차 값이 포함된다면 강외생성이 깨지게 됩니다. 따라서 1차 차분의 방식을 사용할 경우, 종속변수의 시차 값이 포함되면 안됩니다. 만약 강외생성 가정이 만족되지 않는다면, 더 많은 시간에 따라 변하는 설명변수를 포함하여 문제를 완화할 수 있습니다. 
$$
\Delta y_i = \delta_0 + \beta_1 \Delta x_i + \Delta u_i
$$


차분한 회귀식을 이용하여, 강외생성 가정을 만족한 OLS 추정치를 **first-differenced estimator**이라고 부릅니다.



##### Variation in the difference of independent variable

1차 차분이 유효하려면, 설명변수가  관측 단위 i 사이에서 충분한 변동성을 가져야합니다. 
$$
\Delta y_i = \delta_0 + \beta_1 \Delta x_i + \Delta u_i
$$
만약 설명변수가 성별과 같이 더미변수인 경우 1차 차분 후 변수 자체가 완전히 사라질 수 있습니다. 만약, 설명변수가 모든 개체에서 동일한 방식으로 변한다면, 설명변수와 시간 효과가 완전히 결합되면서 다중공선성 문제가 발생할 수 있습니다. 이 경우, 분석을 진행할 때 개별 효과를 분리하는 것이 어려워지게 됩니다.



##### The homoskedasticity assumption

등분산성 가정은 **오차항의 분산이 모든 관측치에 대해 동일해야 한다**는 것을 의미합니다.
$$
Var( \Delta u_i \ | \ X_i)=\sigma^2
$$
이 가정은 OLS 추정량이 최적성을 갖기 위한 중요한 조건 중 하나입니다.

등분산성이 만족될 경우, OLS 추정량은 **최소분산 성질(BLUE, Best Linear Unbiased Estimator)**을 가지며, 효율적인 추정이 가능합니다. 설령 등분산성 가정이 성립하지 않더라도, 우리는 이분산성을 검정하고 수정할 수 있습니다.



#### Two-Period Panel Data : More Independent Variable

$$
y_{it} = \beta_0 + \delta_0 d2_t + \beta_1 x_{it1} + \beta_2 x_{it2} + \dots + \beta_k x_{itk} + a_i + u_{it} \\
i : \text{cross-sectional observation number} \\
t : \text{time period} \\
k : \text{variable label}
$$

여러 개의 설명 변수를 추가해도 패널 데이터 분석을 수행하는데 문제가 없습니다. 



1차 차분을 적용하면,
$$
\Delta y_i = \delta_0 + \beta_1 \Delta x_{i1} + \beta_2 \Delta x_{i2} + \dots + \beta_k \Delta x_{ik} + \Delta u_i \\
\Delta y_i : \text{change of dependent variable } \\
\Delta x_i : \text{change of independent variable } \\
\Delta u_i : \text{new error term}
$$
즉, fixed effect가 제거되었기 때문에 설명변수와 fixed effect가 상관관계를 가지더라도 편향된 추정량을 가지지 않습니다.



또한, 패널 데이터를 활용하면 과거의 설명 변수가 현재의 종속 변수에 영향을 미치는 **시차 변수(lagged variables)**를 포함한 분석도 가능합니다. 

시차변수를 포함하면 다음과 같은 형태가 됩니다.
$$
y_{it} = \beta_0 + \delta_0 d2_t + \beta_1 x_{it} + \beta_2 x_{i,t-1} + a_i + u_{it}
$$
그러나 두 기간의 데이터만 이용할 경우, 시차변수를 포함하기 어렵기 때문에, 더 긴 기간의 데이터를 확보해야합니다. 즉, **시차변수**를 분석하려면 **최소 3기간 이상**의 패널 데이터가 필요합니다.



#### Two-Period Panel Data : Data Structure

패널 데이터를 저장하는 방식 분석을 수행하는데 중요한 영향을 미칩니다. 저장하는 방식에는 두 가지 방법이 있습니다.

첫번째는 **동일한 개체(단위)의 두 기간 데이터를 연속된 행에 저장**하는 것입니다. 이렇게 하면 1차 차분을 쉽게 계산할 수 있고,  Pooled cross sectional 분석을 수행하는데 용이합니다.

<img src="{{site.url}}\images\2025-03-22-method_panel\data1.PNG" alt="data1" style="zoom: 75%;" />



두번째는 **각 개체에 대해 하나의 행만 유지하고, 각 변수에 대해 두 개의 값을 저장하는 방법**입니다. 이 경우 두 시기(t=1, t=2)에 해당하는 별도의 값을 입력해야합니다. 이 방식은 원본 데이터에서 두 기간을 활용한 Pooled OLS 분석을 수행하기 어렵게 만듭니다.

<img src="{{site.url}}\images\2025-03-22-method_panel\data2.PNG" alt="data2" style="zoom:75%;" />



데이터 분석을 진행할 때 주로 첫번째 방식을 많이 사용합니다. 패널 데이터를 효율적으로 분석하기 위해서는 동일한 개체의 여러 기간 데이터를 개별 행으로 나열하는 것이 일반적으로 더 적절합니다.



#### Two-Period Panel Data : Problems

패널 데이터 분석에는 몇 가지 문제점이 있습니다.

첫째, 패널 데이터를 수집하는 과정은 단일 횡단면 데이터를 수집하는 것보다 더 어렵습니다. 특히, 개별 개인을 대상으로 데이터를 반복적으로 수집해야 하는 경우 응답자의 이탈(dropout)이나 비협조 등의 문제로 인해 데이터가 불완전하게 수집될 가능성이 높습니다. 또한, 동일한 개체를 여러 기간 동안 추적해야 하기 때문에 데이터 구축에 많은 비용과 시간이 소요됩니다.

둘째, 설명 변수가 특정 시점에서는 개체 간에 충분한 변동성을 가질 수 있지만, **동일한 개체를 기준으로 시간에 따른 변화를 분석할 때에는 변동성이 크지 않을** 수 있습니다. 즉, 1차 차분을 수행하면 설명 변수의 변화가 매우 작아질 가능성이 있습니다. 이 경우, 회귀 분석에서 변수의 변동성이 낮아짐에 따라 표준 오차가 증가할 수 있으며, 이는 추정된 계수의 신뢰성을 낮추는 결과를 초래할 수 있습니다.

이러한 문제를 해결하는 한 가지 방법은 더 많은 개체를 포함하여 **표본 크기를 증가**시키는 것입니다. 대규모 표본을 활용하면 설명 변수의 변동성이 유지될 가능성이 높아져 표준 오차를 줄일 수 있습니다. 그러나 현실적으로 항상 대규모 패널 데이터를 확보하는 것이 가능하지는 않습니다. 또 다른 해결 방법은 연도별 차분(year-to-year difference) 대신 **더 긴 기간에 걸친 차분(longer differencing)**을 사용하는 것도 유용합니다. 예를 들어, 1년 단위의 차분보다는 5년 단위의 차분을 이용하면 변수의 변화량이 커질 가능성이 높아지고, 이에 따라 추정의 정확성을 높이는 데 도움이 될 수 있습니다.



#### Policy Analysis with Panel Data

패널 데이터는 정책 분석을 할 때 유용합니다. 특히, 두 기간의 패널 데이터가 있는 경우, 특정 프로그램에 참여한 집단을 **처리 집단(treatment group)**, 참여하지 않은 집단인 **통제 집단(control group)**으로 나눌 수 있습니다. 이는 자연실험과 유사하지만, 동일한 개체들이 두 시점 모두에서 관측된다는 차이점이 있습니다.

결과변수와 프로그램 참여 여부를 나타내는 더미변수를 정의하면, 가장 단순한 형태의 모형은 다음과 같습니다.
$$
y_{it}=\ \beta_0 \ +\ \delta_0d2_t \ +\ \beta_1prog_{it} \ +\ a_i \ +\ u_{it}
$$
이 모형에서 a는 개체별로 변하지 않는 고정 효과를 의미하며, 이를 제거하기 위해 1차 차분을 수행해야합니다.
$$
\Delta y_{it}= \delta_0 \ +\ \beta_1 \Delta prog_{i} \ +\ \Delta u_{i}
$$
**고정효과를 제거**하기 위해서는 **더미 변수(prog)도 차분**해야한다는 것입니다. 



만약 프로그램 참여가 두 번째 기간(t=2)에만 이루어진 경우, 차분된 회귀식에서의 OLS 추정량은 단순한 형태를 가집니다.
$$
\hat{\beta_1}= \bar y_{treat} - \bar y_{control}
$$
즉, 두 기간 동안 처리 집단의 평균 변화량과 통제 집단의 평균 변화량의 차이로 계산되비다. 이는 횡단면 데이터에서 사용되는 이중차분법의 패널 데이터 적용에 해당됩니다. 패널 데이터를 이용하는 장점 중 하나는 동일한 개체에 대해 시간 차이를 고려하여 y를 차분하면서 개별 개인, 기업, 혹은 도시 등의 특성을 통제할 수 있습니다. 

만약, 프로그램 참여가 두 기간 모두 발생한 경우, OLS 추정량이 단순하게 나타낼 수 없지만, 추정량은 프로그램 참여로 인해 y가 변화한 평균값을 의미합니다. 이 경우에는 프로그램 할당과 상관관계를 가질 가능성이 있는 시간에 따라 변하는 요인들을 통제한다고 해도, 주요 결과에는 큰 변화가 없습니다. 이렇게 시간에 따라 변화하는 요인들은 차분해서 식에 포함하면 됩니다. 프로그램 효과가 시간에 따라 변화하는 하는 효과를 반영하기 위해서는 모형에 **interaction dummy variable**을 포함하면 됩니다. 
$$
y_{it}=\ \beta_0 \ +\ \delta_0d2_t \ +\ \beta_1prog_{it} \ + \delta_1d2_t \cdot prog_{it} +\ a_i \ +\ u_{it}
$$
위의 식처럼 정책 효과를 포함하는 변수를 포함시켜주면, 특정 기간에 따라 프로그램의 효과가 달라지는지를 분석할 수 있습니다.





#### Panel Data Methods

---

##### Fixed effects estimation

**Fixed effects estimation : demeaned data**

원래의 모델은 다음과 같습니다.
$$
y_{it}=\beta_1x_{it}+a_i+u_{it} \ \ ,\ \ t=1,2,3,...,T
$$
각 개체마다 위의 회귀식에서 시간에 따라 평균값을 도출하게 됩니다.
$$
\bar y_i=\beta_1\bar x_i+a_i+\bar u_i \\ \\

\bar y_i=\frac{1}{T}\Sigma y_{it} \ \ \ \text{(average for T)}
$$
원래 모델에서 평균 모델을 차분하면,
$$
y_{it}-\bar y_i=\beta_1(x_{it}-\bar x_i)+u_{it}-\bar u_i \ \ ,\ \ t=1,2,3,\cdots,T \\
\text{or} \\

\ddot{y}_{it} = \beta_1 \ddot{x}_{it} + \ddot{u}_{it}, \quad t = 1,2,\cdots,T \\
\\

\ddot y_{it}=y_{it}-\bar y_i \quad \text{(time-demeaned data on y)}
$$


의 형태로 나타나게 됩니다.

이렇게 **time-demeaned data 형태의 식**을 **fixed effects transformation**이라고 하고, **within transformation**이라고도 합니다. 

위의 모델에서는 고정 효과인 a가 사라진 것을 확인할 수 있습니다. 그렇다면, time-demeaned data 형태의 식을 pooled OLS로 추정할 수 있습니다. 이 추정치를 **fixed effect estimator** 또는 **within estimator**이라고 합니다.



**Fixed effects estimation : Between Estimator**

**Between Estimator(집단 간 추정량)**는 패널 데이터 분석에서 집단 간 평균을 이용하여 OLS 추정을 수행하는 방법입니다.
$$
y_{it}=\beta_0\ +\ \beta_1x_{it}+a_i+u_{it} \ \ ,\ \ t=1,2,3,...,T \\
$$
Between Estimator는 **개체별 평균을 사용**하여 **시계열 변화를 제거**하고 **횡단면 데이터의 형태**로 변환횝니다.
$$
\bar y_i=\beta_0+\beta_1\bar x_i+a_i+\bar u_i
$$
이 식에서 고정효과(a)는 남아있지만, 시점 t에 따라 변하는 오차항은 개체별 평균으로 변화됩니다. 이 식을 OLS에 적용했을 때, Between Estimator를 얻을 수 있습니다.

만약, 설명변수의 평균과 고정효과 사이의 상관관계가 존재한다면, 
$$
Cov(\bar x_i,a_i) \neq 0
$$
**편향(bias)**이 발생합니다.



하지만, 
$$
Cov(\ddot x_{it},a_i) = 0
$$
개체별 고정효과와 시간에 따른 변동값이 독립이라면, Random effects 추정량을 사용하는 것이 더 효율적입니다. 

Fixed effect estimation의 경우, 강외생성 가정을 만족한다면 추정치는 불편추정량이 됩니다. 

즉,
$$
E(u_{it}\ |\ X_i,a_i) = 0 \ \ \ \text{for all t and i} \\
\Rightarrow cov(\ddot u_{it}, \ddot x_{itj})=0 \ \ : \text{consistent} 
\Rightarrow E(\ddot u_{it}|X)=0 : \text{unbiased}
$$
위의 관계가 성립합니다. 



이 식은 기존의 demeaned data equation에서 설명변수의 개수를 확장하여 나타낸 식입니다.
$$
\ddot{y}_{it} = \beta_1 \ddot{x}_{it1}  + \beta_2 \ddot{x}_{it2} + \beta_3 \ddot{x}_{it3} + \cdots + \beta_k \ddot{x}_{itk} + \ddot{u}_{it}, \quad t = 1,2,\cdots,T \\
i : \text{num of individual} \\
t : \text{num of observation}
$$
여기서 N은 개체수, T는 각 개체별로 시간적 관측 개수를 의미합니다.



 위의 식 같은 경우, 총 NT개의 관측값과 k개의 독립변수를 바탕으로 회귀분석을 진행합니다.

일반적인 경우, OLS의 자유도는 
$$
df=NT-k
$$
로 표현됩니다.

패널 데이터에서는 고정효과를 제거하기 위해서 개체별 평균을 빼는 과정을 수행합니다. 이 과정에서는 개체별로 하나의 자유도를 읽게 되는데, 고정효과를 제거한 후의 자유도는 다음과 같습니다.
$$
df=N(T-1)-k
$$



패널 데이터에서 연도별 영향을 고려하기 위해 **연도 더미 변수(year dummy variables)**을 포함하는 경우가 있습니다. 보통 첫번째 연도를 기준으로 잡고 나머지 연도에 대한 더미 변수를 포함하는데, 원래 회귀식에 시간에 따라 일정한 변화를 보이는 변수가 존재할 경우, 그 **변수의 효과를 개별적으로 식별하기 어렵**습니다. 일반 회귀식에서 시간에 따라 일정한 변화를 보이는 변수와 함께 연도 더미 변수가 존재할 경우, **다중공선성 문제**가 발생하게 되어 원래 설명변수의 계수를 제대로 추정할 수 없게 됩니다. 



##### Random effects Models

$$
y_{it}=\beta_0+\beta_1x_{it1}+\cdots+\beta_kx_{itk}+ a_i+u_{it}
$$

랜덤효과 모형(Random effect, RE)에서는 고정효과 모형과 달리 절편을 포함하여 분석을 진행합니다. 이 모형에서는 개체별 효과가 확률 변수로 가정됩니다.
$$
E(a_i)=0
$$
이는 **개체별 효과의 평균이 0이라는 가정**을 하고, 시간 더미를 설명변수에 포함하는 것이 일반적입니다. 랜덤효과 모형을 사용하려면 원래의 Pooled OLS과 고정효과 모형과는 달리 하나의 가정이 추가됩니다. 

본래의 모형에서는 
$$
E(u_{it}|X_i,a_i)=0 \ \ \ \text{for all t and i}
$$
와 같이 개체별 효과와 설명변수가 주어졌을 때 잔차항의 기댓값이 0이어야 한다는 가정이 존재했고, 이 가정을 만족하면 설명변수와 오차항이 독립적이므로 일반적인 OLS와 유사한 방법으로 추정이 가능해집니다. 

그러나 랜덤효과 모형의 경우
$$
E(a_i|X_i)=0 \ \ \ \text{for all i} \\
\text{where} \ X_i={\{x_{itj}}\}, \ t=1,2,\cdots,T \ ; \ j=1,2,\cdots,k
$$
의 가정이 추가되는데, 이는 **설명변수와 개체별 효과가 모두 독립**이어야함을 의미합니다. 

아래와 같은 방식으로 가정을 표현할 수 있습니다.
$$
Cov(x_{itj},a_i)=0  \\  t=1,2,\cdots,T \ ; \ j=1,2,\cdots,k
$$
이 가정이 만족되지 않으면, 랜덤효과 모형은 사용할 수 없고, **고정효과(Fixed effect) 또는 차분(First Difference) 방법**을 사용해야합니다.

앞에서 오류항을 정의한 것과 같이, 
$$
y_{it}=\beta_0+\beta_1x_{it1}+\cdots+\beta_kx_{itk}+v_{it} \\
v_{it}=a_i+u_{it}
$$
전체 오류항(v)는 개체별 효과(a)와 개별 오차(u)로 구성됩니다. 개체별 효과가 전체 오류항에 포함되기 때문에, 시점 간 상관관계를 가지게 됩니다. (**Serial correlation**이 발생하게 됩니다.)
$$
Corr(v_{it},v_{is})=\frac{\sigma_a^2}{\sigma_a^2+\sigma_u^2}, \ \ \ t \neq s \\
\text{where} \ \sigma_a^2=Var(a_i) \ \ \ \text{and} \ \ \ \sigma_u^2=Var(u_{it})
$$
로 표현할 수 있는데, 개체별 효과의 분산이 클수록 오차 간 상관관계가 커집니다. 만약, 이 상관관계를 무시하고 분석하면, Pooled OLS standard error가 왜곡되므로, **GLS(Generalized Least Squares)**를 사용해야합니다.



위와 같은 Serial correlation 문제를 해결하기 위해 GLS 변환을 수행해야합니다. GLS 변환을 하기 위해서는 **표본의 개수는 크고, 상대적으로 적은 시점의 데이터**를 가지고 있어야합니다. 또한, **unbalanced panels**로 확장될 수 있지만, **balanced panel**라고 가정하고 분석해야합니다.


$$
\lambda=\ 1- \sqrt{\frac{\sigma_u^2}{\sigma_u^2+T\sigma_a^2} }
$$
**λ는 조정 계수(Weight Factor)로,  0과 1 사이의 값**을 가집니다. 이를 해석하면, T가 증가하면 분모가 커지고, 전체 분수가 작아지므로 λ 값이 증가합니다.
$$
T \rightarrow ∞ \ \ , \  \lambda \rightarrow 1
$$

**T가 무한대로 향하면 λ이 1**로 가게 됩니다. 즉, 측정의 신뢰도가 높아집니다. 반대로 T가 작으면 λ값이 작아져 신뢰도가 낮아질 수 있습니다. 

**λ=0이면 Pooled OLS와 동일**하고, **λ=1이면 고정효과(Fixed effect) 모형과 동일**합니다. λ값에 따라 RE 모형이 FE 모형에 가까워지거나 OLS 모형에 가까워집니다. 

랜덤 효과 모형은 먼저 **GLS(Genaralized Least Squares)**를 이용한 변환을 수행한 후, OLS를 적용하는 방식으로 이루어집니다. 이를 위해 개체별 효과의 분산과 개별 오차의 분산을 추정하는 과정이 필요합니다. 분산 추정은 **Pooled OLS 또는 Fixed effect 모형의 잔차를 이용하여 구할 수 있**습니다. 이후 GLS 변환을 적용하는 과정에서 **설명변수와 종속변수에 Quasi-Demeaning 변환을 실시**하며, 변환된 모형에 대해 OLS를 수행하여 최종적인 추정치를 도출하게 됩니다. 

이를 식으로 나타내면,
$$
y_{it} - \lambda \bar{y}_i = \beta_0 (1 - \lambda) + \beta_1 (x_{it1} - \lambda \bar{x}_{i1}) + \cdots 
+ \beta_k (x_{itk} - \lambda \bar{x}_{ik}) + (v_{it} - \lambda \bar{v}_i)
$$
여기서는 **서로 다른 시점 간 오차들 사이에 상관관계가 없음을 의미**하므로 Pooled OLS와 같은 형태로 간단하게 분석할 수 있습니다. 위에서 나타난 식의 설명변수들은 모든 시점에서 일정한 값을 가지게 됩니다.

λ(weight factor, 조정계수)는 실제로 알 수 없지만 추정할 수 있습니다.
$$
\hat{\lambda}=1-\sqrt{\frac{1}{1+T(\hat\sigma_a^2/\hat\sigma_u^2)}} \\
\hat\sigma_a^2  \ \ \text{and} \ \  \hat\sigma_u^2 \ \ \text{are consistent estimator of } \ \ \sigma_a^2 \ \ \text{and} \ \ \sigma_u^2, \ \ \text{respectively}
$$
이 추정량들은 Pooled OLS나 Fixed effect의 잔차항에 기반하여 추정할 수 있습니다.



랜덤효과 모형의 적절성을 판단하기 위해서는 개체별 효과가 설명변수와 상관관계가 존재하는지 검토해야합니다. 이를 확인하는 대표적인 방법이 **Hausman Test(하우스만 검정)**입니다. 하우스만 검정의 **귀무가설**은 개체별 효과가 설명변수와 상관관계가 없으므로 **랜덤효과 모형이 적절하다**는 것입니다. **대립가설**은 개체별 효과가 설명변수와 상관관계가 존재하여 **고정효과 모형이 더 적절하다**는 것을 의미합니다.

검정 결과, p-value가 0.05보다 크면 귀무가설을 기각할 수 없으므로 랜덤효과 모형을 사용할 수 있습니다. 반대로, p-value가 0.05 이하라면 개체별 효과와 설명변수가 상관관계가 있다고 판단하여 고정효과 모형을 사용하는 것이 더 적절합니다.
