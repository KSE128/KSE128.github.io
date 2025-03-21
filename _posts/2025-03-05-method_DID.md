---
published: true
layout: single
title:  "DID(Difference in Differences, 이중차분법)"
categories: Economics
tag: Activity
toc: true
use_math: true
---

이 포스팅은 **DID(Difference in Differences, 이중차분법)**에 대한 내용을 담고 있습니다.

___



## DID(Difference in Differences, 이중차분법)



#### DID란 무엇인가?

어떤 정책이 실제로 효과가 있는지 궁금할 때, 그저 정책 전후만 비교하면 정책 효과와 더불어서 시간이 지나면서 자연스럽게 변한 시간 효과까지 포함하게 됩니다.

우리는 이때 정책에 실제로 효과가 있는지 알아보기 위해 **DID**(Difference in Differences, 이중차분법)를 사용하게 되는 것이지요.

정책 효과를 제대로 확인하기 위해서는 정책의 영향을 받는 그룹, **treatment group(처리 집단)**과 정책의 영향을 받지 않은 그룹, **control group(통제 집단)**을 비교해야합니다.

집단을 비교할 때는 한 번의 비교가 아니라, 정책 전후로 각각 비교하고, 그 변화의 차이를 보는 방식을 **DID**라고 할 수 있습니다.



#### DID의 기본 가정

인과관계를 추정하기 위해서는 몇 가지 기본 가정을 성립해야합니다.

1. **Exchangeability (교환성 가정)**

   처리군과 통제군이 본질적으로 유사해야합니다. 

   교환성을 만족할 경우, 정책의 영향 여부와 상관없이 두 집단은 결과에 영향을 미치는 다른 모든 조건이 동일하다고 가정할 수 있습니다.

   

2. **Positivity (양의 확률 가정)**

   비교하고자 하는 두 집단(처리 집단과 통제 집단)은 정책을 받을 수도 있고, 안 받을 수도 있는 집단이어야 합니다.

   

3. **Stable Unit Treatment Value Assumption (SUTVA)**

   동일한 정책에 영향을 받았다면, 이는 모든 집단 내의 사람에게 동일한 방식으로 적용되어야 합니다.

   스필오버 효과(Spillover Effect) 가 존재하면 안됩니다. 어떤 사람이 정책에 영향을 받았는지가 다른 사람의 결과에 영향을 주면 안됩니다.

   

4. **다른 가정들**

   1. 정책이 어떤 집단에 적용될 때, 그 집단의 결과 변수는 정책 개입 전에 영향을 받은 상태면 안됩니다.

   2. 정책 개입 전에는, 처리 집단과 통제 집단의 결과 변수는 변화 추세가 비슷해야합니다. 
      **(Parallel Trend Assumption)**





#### DID의 원리

그림을 통해 DID의 원리를 더 알아보겠습니다.

---

![didgraph]({{site.url}}\images\2025-03-05-Difference_in_Differences\didgraph.png)

(Pre-intervention : 정책 전, Post-intervention : 정책 후)

(intervention group : 처리집단, comparison group : 통제집단)

분석을 위해 4개의 집단으로 나누어 비교할 수 있습니다.

첫번째 집단 : 정책 전 처리 집단 , 두번째 집단 : 정책 후 처리 집단                    (빨간 선)

세번째 집단 : 정책 전 통제 집단 , 네번째 집단 : 정책 후 통제 집단                    (초록 선)

---



1. 정책 전 (Pre-intervention)

   정책 시행 전에는 처리 집단과 통제 집단은 서로 **상수 차이(Constant difference)**를 가지게 됩니다. 이 차이는 두 집단 사이의 구조적인 차이에서 비롯된 것으로, 이는 시간에 따라 일정하게 유지된다고 가정합니다. **(Parallel Trend Assumption)**



2. 정책 후 (Post-intervention)

   정책이 시행된 이후, 처리 집단과 통제 집단의 결과값 변화 추세를 비교했을 때, 

   **정책의 효과가 없다면** 처리 집단의 변화 추세는 통제 집단과 동일할 것입니다. 이는 상수 차이가 유지됨을 알 수 있습니다. 그림의 빨간 점선을 통해 알 수 있습니다.

   

   **정책의 효과가 존재한다면**, 처리 집단의 변화는 통제 집단과 다르게 나타납니다. 그림의 빨간 실선에서 볼 수 있듯이 처리 집단의 결과값은 통제 집단보다 더 크게 증가하게 됩니다. 추가적인 증가분을 **정책 효과(intervention effect)**라고 합니다.



#### DID 모형

DID를 간단한 회귀식으로 나타내면 다음과 같습니다.
$$
y = \beta_0 + \delta_0 d2 + \beta_1 dT + \delta_1 d2 \cdot dT + \text{other factors} + u \\
$$

$$
y : \text{the outcome variable of interest} \\
dT : \text{treatment group} = 1 \ ,\ \text{control group} = 0 \\
d2 : \text{second time period} = 1 \ ,\ \text{first time period} = 0 \\
\delta_1 : \text{average treatment effect (policy effect)} \\
\text{other factors} : \text{except of the policy effect}
$$



여기서 처리 집단의 유무와 정책 변화 전후는 더미변수로 처리하게 됩니다. 



우리가 관심 가져야할 계수는 d2와 dT의 상호작용항입니다. 

이 추정량이 DID 모형의 핵심이라고 할 수 있습니다.


$$
\hat{\delta_1} = (\bar{y}_{2,T} \ - \bar{y}_{2,C}) \ - \ (\bar{y}_{1,T} \ - \bar{y}_{1,C}) \\
\ \ \ \ =(\bar{y}_{2,T} \ - \bar{y}_{1,T}) \ - \ (\bar{y}_{2,C} \ - \bar{y}_{1,C}) \\
$$

$$
C : \text{control group}   \\
T : \text{treatment group} \\
1 : \text{before policy changes} \\
2 : \text{after policy changes} \\
\hat{\delta_1} = \text{difference-in-differences estimator} \\
$$

DID의 추정량은 처리 집단의 특정 정책 발생 이후와 이전의 결과 변수 평균값의 차이에서 통제 집단의 특정 정책 발생 이후와 이전의 결과 변수 평균값의 차이를 빼면서 특정 정책의 효과를 추정할 수 있게 됩니다.



아래의 표는 4개의 집단 간 결과의 차이를 나타내고 있습니다.

![did sum]({{site.url}}\images\2025-03-05-Difference_in_Differences\did sum.PNG)



#### 참고 자료

1. 서울시립대학교 최승문 교수님 고급계량연습 Chapter 13의 Lecture Note
2. <https://www.publichealth.columbia.edu/research/population-health-methods/difference-difference-estimation>
3. <https://www.e-jcr.org/archive/view_article?pid=jcr-29-4-121>
