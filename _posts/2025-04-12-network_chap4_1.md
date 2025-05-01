### Chapter 4 Descriptive Analysis of Network Graph Characteristics (Part 1)



#### Introduction

많은 복잡한 시스템에서는 핵심 분석 질문들은 네트워크 그래프의 구조나 특성에 대한 질문으로 바꿔서 생각할 수 있습니다. 이 접근은 현실 세계 시스템을 모델링하고 분석할 수 있는 강력한 framework를 제공합니다.

**Social Dynamics (사회적 상호작용)** 

: 세 노드로 구성된 삼합 구조(triads)를 통해 특정 관계 패턴을 분석할 수 있습니다.

**Information or Commodity Flow (정보 또는 상품의 흐름)**

: 그래프 내에서의 경로와 흐름을 모델링하며 분석할 수 있습니다.

**Importance of System Elements (시스템 구성 요소의 중요성)**

: **노드의 중심성(centrality) 지표**를 통해 각 구성 요소의 **상대적 중요도를 파악**할 수 있습니다.

**Community Detection** **(커뮤니티 탐지)**

: 그래프를 적절히 분할(partitioning)함으로써, 내부적으로 밀접한 집단을 식별할 수 있습니다.



#### Vertex and Edge Characteristics

##### Reformulating Questions Using Network Graphs

네트워크 그래프의 기본 구성 요소는 **vertex(정점)**과 **edge(간선)**입니다. 이 구성요소를 분석하기 위해 다양한 metrics가 개발되어 있으며, 주요 지표는 **정점의 차수(vertex degree)** 기반 지표와 **중심성(centrality)** 기반 지표로 나뉩니다. 중심성 기반 지표는 vertex의  성질을 다루게 됩니다.



##### Vertex Degree (정점 차수)

**정점의 차수(degree)**는 특정 정점 v에 연결된 edge의 개수입니다. 

**차수 분포(degree distribution)**는 각 차수를 가지는 정점의 비율을 나타냅니다.

차수의 개념은 가중치 그래프에서 일반화 될 수 있습니다. 정점의 **세기(strength)**는 가중 그래프에서 특정 정점에 연결된 간선들의 **가중치 합계**입니다. 



karate 데이터를 이용해서 확인해봅시다.

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\karate1.PNG" alt="karate1"  />

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\karate.PNG" alt="karate"  />

```R
> degree(karate)
   Mr Hi  Actor 2  Actor 3  Actor 4  Actor 5  Actor 6  Actor 7 
      16        9       10        6        3        4        4 
 Actor 8  Actor 9 Actor 10 Actor 11 Actor 12 Actor 13 Actor 14 
       4        5        2        3        1        2        5 
Actor 15 Actor 16 Actor 17 Actor 18 Actor 19 Actor 20 Actor 21 
       2        2        2        2        2        3        2 
Actor 22 Actor 23 Actor 24 Actor 25 Actor 26 Actor 27 Actor 28 
       2        2        5        3        3        2        4 
Actor 29 Actor 30 Actor 31 Actor 32 Actor 33   John A 
       3        4        4        6       12       17 
> strength(karate)
   Mr Hi  Actor 2  Actor 3  Actor 4  Actor 5  Actor 6  Actor 7 
      42       29       33       18        8       14       13 
 Actor 8  Actor 9 Actor 10 Actor 11 Actor 12 Actor 13 Actor 14 
      13       17        3        8        3        4       17 
Actor 15 Actor 16 Actor 17 Actor 18 Actor 19 Actor 20 Actor 21 
       5        7        6        3        3        5        4 
Actor 22 Actor 23 Actor 24 Actor 25 Actor 26 Actor 27 Actor 28 
       4        5       21        7       14        6       13 
Actor 29 Actor 30 Actor 31 Actor 32 Actor 33   John A 
       6       13       11       21       38       48 
```

쉽게 이해했을 때, degree는 연결 개수, strength는 연결의 '강도'의 총합(얼마나 강하게 연결되어있는가)이라고 생각합니다!

![degree_strength](C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\degree_strength.PNG)

degree를 히스토그램으로 나타내면, 총 3개의 그룹으로 나뉩니다. degree가 낮은 그룹은 0에 가까운 왼쪽에 위치해있고, degree가 높은 그룹은 오른쪽에 위치해있습니다. karate 데이터의 경우, Actor 1과 34가 가장 많은 연결을 가진 노드입니다. 또, Actor 2,3,33은 Actor 1과 34와 가까운 관계를 가지고 있으며, 일정한 연결을 통해 네트워크에서 중요한 역할을 하고 있습니다.



yeast 데이터를 이용하여 확인해봅시다!

```R
> data(yeast)
> ecount(yeast) # edge  
[1] 11855
> vcount(yeast) # vertex  
[1] 2617  
> d.yeast <- degree(yeast)
> # degree distribution
> hist(d.yeast, col="blue",
+      xlab="Degree", ylab="Frequency",
+      main="Degree Distribution")
> # log-log degree distribution
> dd.yeast <- degree.distribution(yeast)
> d <- 1:max(d.yeast) - 1
> ind <- (dd.yeast != 0)
> plot(d[ind], dd.yeast[ind], log="xy", col="blue",
+      xlab="Log-Degree", ylab="Log-Intensity",
+      main="Log-Log Degree Distribution")
```

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\yeast.PNG" alt="yeast" style="zoom:80%;" />

degree distribution의 그래프에서 노란색으로 칠한 부분만 강하게 연결된 것을 알 수 있습니다. 그외의 부분에서는 잔잔하게 연결되어있습니다. karate 데이터보다 분포가 더 다양합니다.



##### Average Neighbor Degree(평균 이웃 차수)

Average Neighbor Degree는 어떤 정점에 연결된 이웃 정점들의 평균 차수를 나타냅니다. 이는, 단순히 어떤 정점이 얼마나 많은 연결을 갖고 있는지만 보는 것이 아니라 **그 정점이 연결된 다른 정점들은 얼마나 연결이 잘 되어 있는지를 확인**하면서 네트워크의 구조적 특성을 이해할 수 있습니다. 

```R
> a.nn.deg.yeast <- knn(yeast, V(yeast))$knn
> plot(d.yeast, a.nn.deg.yeast, log = "xy",
+      col = "goldenrod", xlab = "Log Vertex Degree",
+      ylab = "Log Average Neighbor Degree")
```

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\average_neighbor_degree.PNG" alt="average_neighbor_degree" style="zoom:60%;" />

교수님 설명 추가 !! 

내 degree가 높으면, neighbor degree도 높다. (내 degree가 높다면, 높은 애들끼리 많이 연결되어 있다.) 

반대로 내 degree가 낮으면, neighbor degree도 낮다. (degree가 낮으면, 그 애들끼리 또 따닥 붙어있다.)

물론 아닌 경우도 존재하다!!



##### Types of Centrality

**Degree centrality** 

정점에 연결된 edge의 수로 중심성을 평가합니다. 연결이 많을수록 central하다고 합니다.



**Closeness centrality(근접 중심성)**

한 정점이 다른 모든 정점들과 얼마나 가까운지를 나타내는 지표입니다. 네트워크에서 많은 다른 정점들과 가까운 정점일수록 중심적이다는 아이디어를 반영합니다.
$$
\text{In graph,} \ \ G=(V,E) \\
c_{Cl}(v) = \frac{1}{\sum_{u \in V} \text{dist}(v, u)}
$$
여기서 dist(v,u)는 정점 v와 u 사이의 최단 거리를 의미합니다. v 입장에서 자기를 제외한 모든 vertex와의 거리를 고려해야 합니다. 합이 크면 클수록 centrality가 작아집니다. 거리가 짧을수록 중심성이 높다고 판단합니다. 계산한 결과, 즉 c_CI(v)가 크면 클수록 centrality가 커지게 됩니다.

정규화된 형태로 나타내면 다음과 같은 형태가 나타납니다.
$$
c^{\text{norm}}_{Cl}(v) = \frac{N_v - 1}{\sum_{u \in V} \text{dist}(v, u)} \\
N_v = |V| \ :\text{the total number of vertices in the graph}
$$
정규화된 형태는 0과 1 사이의 값을 가지게 됩니다. 

또, 
$$
N_v-1= \text{max}(\sum_{u \in V} \text{dist}(v, u))
$$
로 이해할 수 있습니다.



**Betweenness centrality(중개 중심성)**

중개 중심성은 어떤 정점이 다른 정점 쌍 간의 최단 경로에 얼마나 자주 위치하는지를 측정합니다. 즉, 네트워크에서 정보 흐르름을 이어주는 '중개자' 역할을 얼마나 잘하고 있는지를 나타냅니다.
$$
G=(V,E) \\
C_B(v) = \sum_{s \ne t \ne v \in V } \frac{\sigma(s, t \mid v)}{\sigma(s, t)}
$$
**σ(s,t)** : 정점 s와 t 사이의 **모든 최단 경로의 수**

**σ(s,t∣v)** : 그 중 **정점 v**를 지나가는 최단 경로의 수



Q. **σ(s,t∣v) = 0** 이 될 수 있는가? 

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\σ(s,t∣v)=0.png" alt="σ(s,t∣v)=0" style="zoom: 80%;" />



가능합니다!



위의 betweenness centrality 계산 방법은 0과 1 사이의 값으로 측정할 수 없습니다. 

그래서, 전체 네트워크 간 비교를 위해 요소들을 나누면 다음과 같습니다.
$$
\frac{(N_v - 1)(N_v - 2)}{2}= \binom{v-1}{2} \\
N_v = |V| \ :\text{the number of vertices}
$$
네트워크의 **다리 역할**을 하는 정점을 강조합니다. 

즉, 여러 노드들 사이의 **최단 경로에 자주 등장**하는 정점일수록 betweenness 중심성이 높습니다.

중심성이 높은 정점은 "**병목 지점(bottlenecks)**" 또는 "**브리지(bridges)**"처럼 작용합니다. 이 정점이 없으면 네트워크의 다른 부분 간 연결이 어렵거나 단절될 수 있습니다.

이런 정점들은 네트워크 내에서의 **정보, 자원, 질병 등의 흐름을 통제하거나 촉진**할 수 있는 **전략적 중요성**을 갖습니다.

따라서 **정보 전파, 질병 확산, 네트워크 취약성 분석** 등에서 매우 유용하게 사용됩니다.



betweenness centrality는 degree centrality와 다른 방향을 지니고 있어 degree centrality는 높지만, betweenness centrality는 낮을 수 있습니다.

예를 들어, 한 노드가 **특정 그룹 내에서는 많은 노드와 연결**되어 있어 **degree centrality는 높**을 수 있지만, 그 노드가 전체 네트워크에서 **다른 그룹 간 연결 고리 역할을 하지 않는다**면, **betweenness centrality는 낮**을 수 있습니다. 



**Eigenvector centrality(고유벡터 중심성)**

Eigenvector centrality는 단순히 얼마나 많은 노드와 연결되어있는지를 넘어서 누구와 연결되어있는가를 중요하게 생각하고 있습니다. 즉, 영향력 있는 노드와 연결되어 있다면, 그 노드 자체도 중요하다는 아이디어에 기반합니다. 


$$
G=(V,E), \ \ \text{adjacency matrix A}, \\
c_{Ei}(v) = \alpha \sum_{\{u, v\} \in E} c_{Ei}(u)
$$


행렬 형태로 다음과 같이 표현할 수 있습니다.
$$
A \cdot c_{Ei} = \alpha^{-1} \cdot c_{Ei} \\
c_{Ei} : \text{Eigenvector corresponding to the maximum eigenvalue of adjacency matrix A}
$$
고유벡터 중심성은 인접행렬 A의 최대 고윳값에 대응하는 고유벡터라고 할 수 있습니다.



**Interpretation**

어떤 노드가 중요한 이유는, 중요한 노드들과 연결되어 있기 때문입니다. 단순히 연결 수(degree)가 아닌, 연결된 상대의 영향력도 함께 고려한다는 의미를 가지고 있습니다.

이웃들의 중심성 정보를 재귀적으로 반영합니다. 내 중심성은 이웃들의 중심성에 따라 정해지고, 그 이웃들의 중심성도 또 그 이웃들의 중심성에 따라 정해지는 방식입니다. 

노드 A의 중심성은 연결된 B, C 등의 중심성에 따라 달라지고, B, C는 또 그들의 이웃 중심성에 따라 달라지니, 서로 영향을 주고받는 구조가 됩니다. 이를 self-referential(자기 지시적), recursive(재귀적)이라고 합니다.

이 중심성 개념은 영향력 전파 모델, 선형 모델들에서 자연스럽게 등장합니다.



**Properties**

연결된 무방향 그래프라면, 고유벡터 중심성은 **음수가 없고 모두 양수 혹은 0**입니다. 모든 노드는 중심성 수치가 **명확하게** 정의됩니다. 또한, 계산된 고유벡터 중심성은 **절대적인 수치가 아니라 상대적 중요도**이기 때문에,
 일반적으로 **가장 큰 값을 1로 정규화**해서 해석을 쉽게 합니다.



##### Caculating the Centrality of Graph

Example Graph 1

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\example_graph1.PNG" alt="example_graph1" style="zoom: 67%;" />

```R
> edges <- c("A", "B", "B", "C", "B", "D", "C", "G", "C", "E", "E", "F")
> g <- graph(edges, directed = FALSE)
> 
> closeness_cen <- round(closeness(g, normalized = TRUE), 3)
> betweenness_cen <- round(betweenness(g, normalized = TRUE), 3)
> eigenvector_cen <- round(eigen_centrality(g)$vector, 3)
> 
> centrality_table <- data.frame(
+   Vertex = V(g)$name,
+   Closeness = closeness_cen,
+   Betweenness = betweenness_cen,
+   Eigenvector = eigenvector_cen
+ )
```

```R
> print(centrality_table)
  Vertex Closeness Betweenness Eigenvector
A      A     0.400       0.000       0.452
B      B     0.600       0.600       0.927
C      C     
D      D     0.400       0.000       0.452
G      G     0.429       0.000       0.487
E      E     0.500       0.333       0.639
F      F     0.353       0.000       0.311
```



vetex C의 Centrality 를 구해봅시다.

**Closeness Centrality**
$$
c^{\text{norm}}_{Cl}(v) = \frac{N_v - 1}{\sum_{u \in V} \text{dist}(v, u)} \\
N_v = |V| \ :\text{the total number of vertices in the graph}
$$
와 같습니다. 자기 자신을 제외한 모든 노드들과의 거리를 합한 후 공식에 대입합니다.
$$
\frac{N_v(7)-1}{2+1+2+1+2+1}=\frac{6}{9}=0.667
$$
분모에서 합하는 순서는 A와 C 사이의 거리, B와 C 사이의 거리, D와 C 사이의 거리 ... 로 이루어져 있습니다.

정규화한 공식에 따르면, Closeness Centrality는 0.667이 됩니다.



**Betweenness Centrality**
$$
\sum_{s \ne t \ne v \in V } \frac{\sigma(s, t \mid v)}{\sigma(s, t)} \div {\binom{v-1}{2}} \\ \\
N_v = |V| \ :\text{the number of vertices}
$$

| 엣지 | 값   | 엣지 | 값   | 엣지 | 값   | 엣지 | 값   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| AB   | 0    | BD   | 0    | DE   | 1    | EF   | 0    |
| AD   | 0    | BE   | 1    | DF   | 1    | EG   | 1    |
| AE   | 1    | BF   | 1    | DG   | 1    | FG   | 1    |
| AF   | 1    | BG   | 1    |      |      |      |      |
| AG   | 1    |      |      |      |      |      |      |

값을 모두 합해보면, 11이라는 값이 나옵니다. 
$$
\frac{11}{15} \approx 0.7333
$$


Betweenness Centralilty는 0.733의 값을 가집니다.



**Eigenvector centrality**

계산된 고유벡터 중심성은 **절대적인 수치가 아니라 상대적 중요도**이기 때문에, 일반적으로 **가장 큰 값을 1로 정규화**해서 해석을 쉽게 한다는 property에 따라, 위의 그래프에서는 C가 가장 많은 연결을 가지고 있어 Eigenvector Centrality는 1이 됩니다.





Example Graph 2

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\example_graph2.PNG" alt="example_graph2" style="zoom:67%;" />

```R
> g <- graph(edges, directed = FALSE)
> 
> closeness_cen <- round(closeness(g, normalized = TRUE), 3)
> betweenness_cen <- round(betweenness(g, normalized = TRUE), 3)
> eigenvector_cen <- round(eigen_centrality(g)$vector, 3)
> 
> centrality_table <- data.frame(
+   Vertex = V(g)$name,
+   Closeness = closeness_cen,
+   Betweenness = betweenness_cen,
+   Eigenvector = eigenvector_cen
+ )
```

```R
> print(centrality_table)
  Vertex Closeness Betweenness Eigenvector
A      A     0.429       0.000       0.405
B      B     0.667       0.600       1.000
C      C                             0.891
D      D     0.429       0.000       0.405
G      G     0.545       0.000       0.766
E      E     0.500       0.333       0.432
F      F     0.353       0.000       0.175
```

vetex C의 Centrality 를 구해봅시다.

**Closeness Centrality**
$$
\frac{N_v(7)-1}{2+1+2+1+2+1}=\frac{6}{9}=0.667
$$
분모에서 합하는 순서는 A와 C 사이의 거리, B와 C 사이의 거리, D와 C 사이의 거리 ... 로 이루어져 있습니다.

정규화한 공식에 따르면, Closeness Centrality는 0.667이 됩니다.



**Betweenness Centrality**

| 엣지 | 값   | 엣지 | 값   | 엣지 | 값   | 엣지 | 값   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| AB   | 0    | BD   | 0    | DE   | 1    | EF   | 0    |
| AD   | 0    | BE   | 1    | DF   | 1    | EG   | 1    |
| AE   | 1    | BF   | 1    | DG   | 0    | FG   | 1    |
| AF   | 1    | BG   | 0    |      |      |      |      |
| AG   | 0    |      |      |      |      |      |      |

값을 모두 합해보면, 8이라는 값이 나옵니다. 
$$
\frac{8}{15} \approx 0.533
$$


Betweenness Centralilty는 0.533의 값을 가집니다.



**Eigenvector centrality**

여기서 Eigenvector centrality는 어떻게 구하는 것일까용 ~ 나도 몰라 ... R에 따르면 0.891로 나오네용



##### Back to Karate Example

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\karate_back.PNG" alt="karate_back" style="zoom:80%;" />

위의 4가지 그래프는 Karate 데이터의 중심성 그래프를 표현하고 있습니다. 

각각의 centrality measure별로 동심원 이루고 있고, 

같은 동심원 상에 위치한 vertex 값은 모두 같은 centrality 값을 가집니다. 그리고, 동심원의 중심으로 갈수록 centrality 값이 커집니다.

degree centrality의 그래프를 봤을 때, 노란점과 파란점이 동심원의 중심 근처에 있습니다. 나머지 점들은 중심 근처에서 조금 벗어나거나 조금 더 벗어난 형태로 이루어져있는데, 이 모습은 degree centrality의 히스토그램과 비슷한 모습이 보입니다. 히스토그램에서 3개의 그룹으로 보였는데, 잘 도식화되어있습니다.

degree와 가장 큰 차이를 보이는 그래프는 betweenness와 eigenvector centrality 입니다.

betweennes의 경우 파랑값은 확실히 높은 값으로 나타나는데, 노란색은 파란색보다 작아진 값을 가지게 됩니다. degree와 다르게 세 그룹으로 나뉘지는 않지만, 노란점과 파란점은 높은 값으로 한 그룹, 나머지 점들을 다른 그룹으로 바뀌게 됩니다.

eigenvector의 경우, 노란색 주위에 여러 연결이 많은데, 얼마나 주위에 인플루언서가 많은지를 확인할 수 있습니다.



##### HITS

위에서 언급한 Degree, Cloneness, Closeness, Betweenness 등 중심성 개념은 무방향 그래프에서 사용되지만, 방향성이 있는 경우에도 자연스럽게 확장이 가능합니다.

추가로 HITS(Hyperlink-Induced Topic Search) 알고리즘에서는 방향성이 있는 그래프에서 두 가지 종류의 중요한 노드를 식별합니다. Authorities 점수와 Hubs 점수가 있습니다. 

Authorities 점수는 해당 노드가 얼마나 많은 신회할 수 있는 허브들로부터 링크를 받았는지를 나타냅니다. 좋은 허브들에 의해 많이 참조되는 노드는 권위 있는 노드로 간주됩니다. Hubs 점수는 해당 노드가 얼마나 권위 있는 노드들을 가리키는지를 나타냅니다. 좋은 허브는 좋은 권위 노드들로 연결되어 있어야합니다. 이 구조는 순환참조의 루프라고 할 수 있습니다. 이러한 구조는 상호 보완적인 방식으로 작동합니다. 즉, 좋은 허브는 좋은 권위를 가리키고, 좋은 권위는 좋은 허브에 의해 가리켜진다는 점에서 서로의 점수를 강화하게 됩니다.

웹 검색의 맥락에서 보면, 어떤 웹페이지는 여러 유용한 링크들을 모아놓은 출발점(허브) 역할을 하고, 어떤 웹페이지는 신뢰할 수 있는 정보의 목적지 역할을 합니다. HITS 알고리즘은 이러한 역할의 구분을 통해, 정보의 흐름에서 중요한 노드들을 효과적으로 식별할 수 있습니다.



**HITS 알고리즘 핵심 아이디어**
$$
\text{Authority Score : } a_i \\
\text{Hub Score : } h_i
$$


Scores는 재귀적으로 정의됩니다.


$$
a_i = \sum_{j: j \to i} h_j \\
$$


: 나를 가리키는 노드들의 허브 점수 합


$$
h_i = \sum_{j: i \to j} h_j
$$
: 내가 가리키는 노드들의 권위 점수 합



**좋은 Authority**는 **좋은 Hub들로부터** 연결되고, **좋은 Hub**는 **좋은 Authority들에게** 연결됩니다.

이 구조는 서로를 강화하는 **상호 순환 구조**라고 할 수 있습니다.



**HITS Matrix Formulation**

A는 방향 그래프(directed graph)의 인접 행렬(adjacency matrix)입니다.


$$
A_{ij} = \begin{cases} 
1 & \text{if } i \to j \\
0 & \text{otherwise}
\end{cases}
$$
만약, 노드 i에서 노드 j로 가는 화살표가 존재한다면 1의 값을 가지고, 그렇지 않다면 0의 값을 가집니다.
$$
\text{Authority scores : Principal eigenvector of }A^TA \\
\text{Hub scores : Principal eigenvector of }AA^T
$$
Authority 점수는 현재의 허브 점수를 바탕으로 계산되며, 
$$
a \leftarrow A^Th
$$
이는 행렬 A의 전치행렬과 허브 점수 벡터 h의 곱으로 구합니다.

반대로, Hubs 점수는 현재의 Authority 점수를 이용해 계산하며, 
$$
h \leftarrow Aa
$$
인접 행렬 A와 Authority 점수 벡터 a의 곱으로 나타납니다.



두 과정을 반복적으로 수행하면, 각 노드에 대해 수렴된 형태의 안정적인 Authority 점수와 Hub 점수를 얻을 수 있습니다.



<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L3\HITS.PNG" alt="HITS" style="zoom:80%;" />

위의 그래프는 HITS(Hyperlink-Induced Topic Search) 알고리즘의 시각적 예시로 웹페이지나 블로그처럼 방향성이 있는 네트워크에서 노드의 중요도를 두 가지 관점, Hubs와 Authorities 관점에서 평가합니다.

왼쪽 그림(Hubs)에서 큰 노드들은 다양한 권위 있는 블로그를 참조하는 블로그들입니다. 이들은 스스로 권위 있지는 않지만, 좋은 정보를 많이 링크하는 "가이드" 같은 역할을 합니다. 오른쪽 그림(Authorities)에서 큰 노드들은 많은 허브 블로그로부터 참조되고 있는 블로그들입니다. 정보의 중심에 있는 권위 있는 블로그라고 할 수 있습니다.



##### Types of Centrality (+ Edge centrality, Edge betweenness centrality)

**Edge Centrality(간선 중심성)**

네트워크 분석은 vertex-based(degree and centrality)으로 이루어집니다.

네트워크에서는 "어떤 사람?"도 중요하지만, "어떤 관계?", "어떤 연결?"이 더 중요할 때가 있습니다.

예를 들어, 정보 확산에 가장 중요한 관계는 무엇인가? 또는 네트워크 내에서 어떤 연결이 중요한 다리 역할을 하는가? 와 같은 질문에서는 vertex가 아니라 edge에 주목해야합니다.



**Edge Betweenness Centrality(간선 매개 중심성)**

Betweenness Centrality에서는 어떤 vertex가 얼마나 많은 최단 경로 위에 있는지를 나타내는 값을 구했는데, 이를 edge에 적용한 것이 Edge Betweenness Centrality 입니다.

Betweenness Centrality: vertex가 얼마나 주요 길목에 잘 많이 놓여 있는가! (shortest path 개념)

Edge Betweenness Centrality은 어떤 edge가 네트워크 내에서 **얼마나 많은 최단 경로(shortest path)**를 지나가는지를 측정한 값입니다. 값이 높을수록, 그 edge는 다른 vertex 간의 경로에서 자주 사용된다는 뜻입니다. 즉, 그 edge가 정보 흐름의 핵심 경로라는 의미입니다. 값이 크면 클수록 더 중요한 edge이고 더 집중되어 있는 edge입니다. 



Karate Club Network의 예시를 확인해봅시다!

```R
> eb <- edge_betweenness(karate)
> E(karate)[order(eb, decreasing = TRUE)[1:3]]  
+ 3/78 edges from 4b458a1 (vertex names):
[1] Actor 20--John A   Mr Hi   --Actor 20
[3] Mr Hi   --Actor 32
```

karate 데이터의 edge betweeenness를 계산해서 상위 3개의 edge를 추출한 결과, 

Actor 20--John A가 가장 높은 중요도를 가지고, 다음으로,Mr Hi --Actor 20가 높은 중요도를 가지고 있다는 것은 당연한 이야기입니다. 

네트워크 관계도를 통해 확인해보면,

![karate1](C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L4\karate1.PNG)

Mr Hi와 John A는 각 그룹의 핵심 인물임을 확인할 수 있습니다. 

그런데 이 두 사람 다 Actor 20과 연결되어 있는 edge가 높은 것을 통해 두 핵심 인물 사이에 중요한 연결고리라는 것을 알 수 있습니다. 즉, Actor 20과 연결된 간선들은 여러 그룹 사이의 다리(bridge) 역할을 하며, 네트워크의 여러 부분 사이의 많은 최단 경로에 포함되어 있습니다.



##### Beyond Betweenness : Edge Centrality via Line Graphs

Betweenness Centrality처럼 경로(path)의 개념이 직접적으로 사용되지 않는 centrality measure의 경우, 다른 centrality 개념을 edge에 적용하기는 어렵습니다. 이는 대부분의 centrality 지표들이 정점(vertex)에 기반하고, 경로 개념이 명확하게 도입되지 않으면 vertex과 edge를 동시에 고려하기 어렵기 때문입니다. 따라서 이러한 개념을 edge 중심으로 확장하여 논의하기 위해서는 line graph를 도입하는 것이 적절합니다.

Line graph는 기존 그래프의 간선(edge)들을 vertex로 바꿔서 새로 만든 그래프입니다.

원래 그래프 G가 있을 때, 그 line graph G' 는 G의 각 edge을 G'의 vertex으로 바꿉니다. 

G에서 같은 vertex에 연결된 두 edge는 그 edge을 vertex로 바꾼 그래프 G'에서 연결됩니다.

---

**쉬운 예시!!**

원래 그래프 G :

A --- B --- C 

간선1 : A --- B, 간선2 : B --- C  → 두 간선은 공통 vertex B를 통해 연결되어 있습니다.

이걸 line graph G'로 바꾸면,

간선1 (A --- B) :  G' 에서 **vertex 1**, 간선2 (B --- C) : G' 에서 **vertex 2**

그런데 A --- B와 B --- C는 **공통 vertex B**를 가졌으니까, G'에서는 vertex 1과 vertex 2가 연결됩니다.

즉, G에서 같은 vertex를 공유한 두 edge가 G'에서는 서로 연결된 노드가 되는 것입니다!!

---

igraph 패키지에서 line.graph(G)를 이용하면, line graph형태로 변환되어 나타납니다. line graph로 변환하면 원래 네트워크의 edge를 vertex처럼 다룰 수 있어서, vertex centrality 지표를 edge 분석에 활용할 수 있습니다.

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L4\line_graph.PNG" alt="line_graph" style="zoom:80%;" />
