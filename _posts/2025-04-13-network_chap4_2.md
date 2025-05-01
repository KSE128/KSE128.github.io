### Chapter 4 Descriptive Analysis of Network Graph Characteristics (Part 2)



#### Characterizing Network Cohesion

##### Network Cohesion : Clique

Network Cohesion(네트워크 응집력)이란, 네트워크의 밀접한 연결 정도를 의미합니다. 이를 확인하는 방법은 특정 구조를 가진 부분 그래프(subgraph)를 찾아보는 것입니다.

가장 고전적인 예시가 **clique(클리크)**라고 하는데,  **모든 vertice 쌍이 서로 연결된 노드 집합**을 의미합니다. 즉, **완전 연결 그래프**입니다. 

**"Conducting a census of subgraphs like cliques provides a structural snapshot of how cohesive a network is."**

클리크(clique)와 같은 부분그래프를 전수 조사하면, 네트워크의 응집력이 얼마나 높은지를 구조적으로 파악할 수 있습니다. → clique는 완전 연결 그래프이기 때문. 



Karate 네트워크 데이터를 이용하여 **clique census**를 파악할 수 있습니다.

: clique census란? 네트워크 내에 존재하는 모든 크기의 clique를 세고 기록하는 과정

```R
> table(sapply(cliques(karate), length))

 1  2  3  4  5 
34 78 45 11  2 
```

size 1 : individual nodes, size 2 : edges, size 3 : triangles

```R
> cliques(karate)[sapply(cliques(karate), length)==5]
[[1]]
+ 5/34 vertices, named, from 4b458a1:
[1] Mr Hi    Actor 2  Actor 3  Actor 4  Actor 14

[[2]]
+ 5/34 vertices, named, from 4b458a1:
[1] Mr Hi   Actor 2 Actor 3 Actor 4 Actor 8
```

두 clique 모두 Mr Hi를 포함하고, 네 명의 구성원이 겹치게 됩니다.

또, Mr Hi가 clique의 중심이었다는 것을 알 수 있습니다.



clique라는 개념에서 **clique number** 개념이 나타납니다. 그래프에서 찾을 수 있는 **가장 큰 clique의 크기**인데, Karate 네트워크에서는 clique number가 5가 됩니다.

```R
> clique_num(karate)
[1] 5
```



clique number가 크면 클수록 그래프의 tightness 측면에서는 tight하게 연결되어있다고 할 수 있습니다.



##### Cliques : Maximum vs. Maximal

**Maximum Clique**는 전체 그래프가 가질 수 있는 가장 큰 크기의 complete clique를 의미합니다. 

```R
> largest <- largest_cliques(karate)
> print(largest)
[[1]]
+ 5/34 vertices, named, from 4b458a1:
[1] Mr Hi    Actor 2  Actor 3  Actor 4  Actor 14

[[2]]
+ 5/34 vertices, named, from 4b458a1:
[1] Mr Hi   Actor 2 Actor 3 Actor 4 Actor 8
```



**Maximal Clique**은 **자기 자신이 clique**이어야하고, **더 큰 clique의 부분집합이면 안된다**는 조건을 가지고 있습니다. 정점 하나를 추가하면, clique 구조가 깨지고, 일부 정점을 빼면 여전히 clique가 될 수 있지만, **그것들은 더 이상 maximal은 아닙니다.**

간단한 예시를 통해 이해해봅시다!

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L4\max_simple_example.png" alt="max_simple_example" style="zoom: 40%;" />

vertex 1,2,3이 서로 연결되어 있고, vertex 4는 1,2와만 연결되어 있있습니다.

| 집합            | clique? | maximal clique? |
| --------------- | ------- | --------------- |
| {1 , 2}         | O       | X               |
| {1, 2 , 3}      | O       | O               |
| {1 , 2 , 4}     | O       | O               |
| {1 , 2 , 3 , 4} | X       | X               |



다시 Karate 데이터를 이용하여 분석해보겠습니다.

```R
> table(sapply(max_cliques(karate), length))

 2  3  4  5 
11 21  2  2 

> table(sapply(cliques(karate), length))

 1  2  3  4  5 
34 78 45 11  2 
```

결과를 비교해보면, cliques()의 개수보다 max_cliques()의 개수가 적은 것으로 나타납니다. 

cliques()는 그래프 내 존재하는 모든 크기의 clique를 탐색합니다. 큰 clique의 부분집합도 전부 포함되는데, max_cliques()는 더 이상 확장할 수 없는 clique만 포함합니다. 즉, subset clique는 제외하고 카운트합니다.

```R
> cliques(karate)[sapply(cliques(karate), length)==5]
[[1]]
+ 5/34 vertices, named, from 4b458a1:
[1] Actor 1    Actor 2  Actor 3  Actor 4  Actor 14

[[2]]
+ 5/34 vertices, named, from 4b458a1:
[1] Actor 1    Actor 2 Actor 3 Actor 4 Actor 8

> table(sapply(max_cliques(karate), length))
 2  3  4  5 
11 21  2  2 
```

위의 결과에서, 

{1 , 2 , 3 , 4}, {1 , 2 , 3 , 8}, {2 , 3 , 4 , 8}은 모두 clique을 이룹니다. 

**Q. {2 , 3 , 4 , 8}은 maximal clique인가?**

{2 , 3 , 4 , 8}은 {1 , 2 , 3 , 4 , 8}의 부분집합(subset)으로 clique은 맞지만, maximal clique은 아닙니다. 

위에 언급한 조건을 봤을 때, 더 큰 clique의 부분집합이면 안된다고 했는데, {2 , 3 , 4 , 8}은 {1 , 2 , 3 , 4 , 8}의 부분집합이기 때문에 조건에 어긋납니다. 즉, maximal clique가 아닙니다.



**Q. 그렇다면, {1 , 2 , 3 , 4 , 8}은 maximal clique인가?**

맞습니다. 무언가의 subset이 아니고 모두 연결되어있기 때문입니다.



##### K-Cores

K-core는 두 가지 조건을 만족하는 subgraph입니다. 먼저, **그래프에 속한 모든 정점의 차수(degree)가 k이상**이고, 이 조건을 만족하는 **maximal한 집합**이어야합니다. 

K-core는 Clique의 **완화된 버전**이라고 합니다. Clique는 모든 노드가 서로 연결되어 있어야합니다. 그러나, K-core는 각 노드가 **적어도 k개 노드랑 연결**되기만 하면 됩니다. 그래서 현실 네트워크 분석에 더 적합합니다.



"K-core decompositions often appear as layers (like onions), "

K-core 분해는 흔히 양파처럼 층을 이루는 구조로 나타납니다. 바깥에서 안쪽으로 들어갈수록 연결의 밀도가 높아지며, 이는 **각 vertex가 최소 k개의 이웃과 연결**되어 있는 K-core로 정의합니다.

- k = 1 코어는 **가장 약하게 연결된 주변부**를 나타내고,
- k = 2, k = 3으로 갈수록 **점점 더 중심적인 위치**로 나아가며,
- **높은** **k** 값을 가지는 코어는 **가장 단단히 연결된 중심 집단을 의미**합니다.



"and are helpful invisualizing core-periphery structure."

이러한 특성 때문에, K-core 분해는 네트워크의 **core-periphery 구조(중심-주변 구조)**를 시각적으로 파악하는 데 매우 유용합니다. 

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L4\kcore.PNG" alt="kcore" style="zoom:50%;" />



**k-core decomposition** 결과를 **동심원 구조**로 시각화한 것으로 양파 껍질처럼 분해해서 노드가 얼마나 중심적인지를 보여주고 있습니다. 노드의 색깔과 위치를 통해 각 노드가 속한 k-core 레벨을 시각적으로 구분할 수 있습니다. 중**심에 가까운 노드일수록 더 높은 k-core**에 속하고 **바깥으로 갈수록 연결성이 낮은 노드**임을 나타냅니다. 또, 같은 k-core 값(coreness)을 가진 노드들이 같은 원 안에 배치됩니다



**교수님께서 언급하신 내용!!**

**k-vertex connected graph**란, 어떤 **정점** k개 미만을 제거해도 그래프가 계속 연결되어 있는 그래프입니다. 정점 연결성을 유지하려면 **최소 몇 개의 정점을 제거해야** 연결이 끊기는지에 대한 질문에서 그 숫자가 k일 때, 그래프는 k-vertex connected 그래프라고 합니다. 예를 들어 어떤 그래프가 2-vertex-connected라면, 아무 정점 1개를 제거해도 그래프가 끊기지 않습니다. 하지만 2개를 제거하면 끊어질 수도 있습니다.

**k-edge connected graph**란, 어떤 **간선** k개 미만을 제거해도 그래프가 계속 연결되어 있는 그래프입니다. 쉽게 말해서, **간선 몇 개를 끊어야** 그래프가 분리되는가라는 질문에서 그 숫자가 k일 때, k-edge connected라고 부릅니다.

**k-star graph**란, 중심 노드가 1개가 있고, 이 중심과 주변 노드들이 각각 간선 1개로만 연결된 구조입니다.

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L4\kstar.PNG" alt="kstar" style="zoom: 67%;" />

여기서 **중심 정점**은 1, **주변 정점**은 2, 3, 4, 5, 6, 7로 각각 간선 하나로 1에 연결됩니다.



**(a) k-vertex-connected 관점**

정점 1 (중심) 하나를 제거하면, 주변 정점들은 아무 연결도 없게 됩니다. 하나 제거하는 순간, 그래프는 완전히 분리됩니다. 따라서 **k-star 그래프**는 **1-vertex-connected-그래프**입니다.



**(b) k-edge-connected 관점**

간선 하나만 제거해도, 해당 간선에 연결된 leaf 노드는 더 이상 중심과 연결되지 않습니다. 예를 들어, edge 1--2을 제거하면, 2는 네트워크에서 고립됩니다. 따라서,  **k-star 그래프**는 **1-edge-connected-그래프**입니다.



**높은 연결성(k가 큰)** 그래프일수록 **강한** 네트워크를 의미합니다.

**k-star 그래프**는 매우 취약한 구조입니다. 중심 노드 하나 또는 하나의 연결만 끊겨도 전체가 붕괴됩니다.



##### Dyads and Triads (in Directed Graphs)

**Dyad**는 정점(vertex) **2개**로 이루어진 **가장 작은 subgraph**입니다. Dyad의 경우, Directed graph에서는 3가지 유형으로 나뉩니다. **Null dyad**는 두 노드 간에 아무런 **연결이 없습니다**. **Asymmetric dyad**는 **한 방향으로만** 연결된 경우를 나타내고, **Mutual dyad**는 **양방향 모두** 연결된 경우를 나타냅니다. 

**Triad**는 정점 **3개**로 구성된 **subgraph**입니다. 방향성과 연결의 **유무**에 따라 총 16가지 가능한 구조가 존재합니다. 

**Census**란, 그래프 내에서 **각 유형이 몇 번 나오는지 세는 것**인데, 이를 통해 네트워크의 **연결 패턴**이나 **상호작용 구조**를 분석할 수 있습니다. Census의 횟수를 통해 **이 네트워크가 얼마나 상호적인가, 중심성이 분산되어 있는가, 혹은 위계적인가** 같은 특징들을 분석할 수 있습니다.

```R
> aidsblog <- upgrade_graph(aidsblog)
> dyad_census(aidsblog)
$mut
[1] 3

$asym
[1] 177

$null
[1] 10405

> triad_census(aidsblog)
 [1] 484621  20717    300   2195     39     74      1    112      4      0      2
[12]      0     15      0      0      0
```



**dyad census**를 확인했을 때, **아무 연결이 없는 null dyads**는 10,405개, **한방향 연결(asymmetric dyads)**는 177개, **상호 연결(mutal dyads)**은 단 3개로 이 네트워크는 대부분 **일방적인 연결**을 하고 있거나 **비연결 상태**입니다. 즉, **상호작용이 매우 제한적**이고, **연결이 일방향적인 성격**이 강합니다.



##### Motifs

Motifs는 그래프(특히 directed graph) 내에서 반복적으로 등장하는 작은 subgraphs 구조를 말합니다. 보통 2~4개 정도의 정점으로 이루어진 자주 나타나는 미니 패턴입니다. 단순히 자주 등장하는게 아니라, 특정 기능적 역할을 하기 때문에 생물학적으로 의미가 있습니다.

Motifs의 예시로는 두 가지가 있는데, **Fan(팬 구조)**로 **하나의 노드가 여러 개의 노드로 일방적으로 연결**되어 있는 구조입니다. 두번째는 **Feedforward Loop로 구조**는 **u → v, v → w, u → w**로 이루어져 있는데, u**가 직접 w에 영향을 주고, 동시에 v를 거쳐 간접적으로도 영향을 주는 이중 경로**를 말합니다.

그런데, Motif를 완전히 다 나열하려면, 모든 노드 조합을 하나하나 확인해야해서 계산량이 매우 큽니다. 그래서 보통 **샘플링 기법**을 이용해서 **일부만 뽑아** 분석합니다.

```R
> graph.motifs(graph, size = 3, sample = 1000)
```



##### Density and Related Notions of Relative Frequency

질문 후 정리





##### Ego-centric Graph

**Ego-centric Graph**는 **특정 vertex 하나(ego)를 중심**으로, 그와 **직접 연결된 이웃(alters)만 포함**하는 **subgraph 네트워크**입니다. Ego-centric Graph의 구성 요소는 **ego, alters, edge**로 이루어져있습니다. **ego**는 분석의 **중심이 되는 노드**입니다. **alters**는 **ego와 직접 연결**된 이웃 노드들이고, **edge**는 **ego와 이웃들 사이의 관계 또는 이웃들끼리의 관계**입니다. ego-centric graph는 특정 노드를 둘러싼 지역 구조를 탐색할 수 있게 해주며, 해당 노드의 즉각적인 환경과 관계에 대한 통찰력을 제공하기 때문에 네트워크 분석에 유용합니다.



**Example : Karate Network에서의 Ego Subgraphs**

전체 네트워크의 밀도와, **정점 1번과 34번**을 중심으로 한 두 개의 ego-centric subgraphs의 밀도 비교:

```R
> ego.instr <- induced_subgraph(karate, neighborhood(karate, 1, 1)[[1]])
> ego.admin <- induced_subgraph(karate, neighborhood(karate, 1, 34)[[1]])
> edge_density(karate)
[1] 0.1390374
> edge_density(ego.instr)
[1] 0.25
> edge_density(ego.admin)
[1] 0.2091503
```



이 ego-네트워크들이 전체 네트워크보다 밀도가 높다는 것은, **중심 인물 주변의 국소적 관계들이 더 밀접하게 연결되어 있다**는 것을 의미합니다.

**karate 네트워크**에서 정점 1번(instr)를 중심으로 한 **ego-graph**과 정점 34번(admin)을 중심으로 한 **ego-graph**

전체 karate 네트워크의 밀도는 **약 0.14**로 그래프 전체에 비해 꽤 희소한 편입니다. 

**Ego-1 (instr)**의 밀도는 0.25이고, **Ego-34 (admin)**의 밀도는 0.21로 나타납니다. 이는 전체 네트워크보다 훨씬 조밀하게 연결되어 있습니다. 즉, 중심 인물 주변의 로컬 네트워크는 연결이 더 강하다는 걸 의미합니다.



##### Global Clustering (Transitivity)

**Clustering Coefficient 또는 Transitivity**는 그래프에서 **삼각형 구조가 얼마나 잘 형성**되는지를 나타내는 지표입니다. 
$$
cl_T(G) = \frac{3 \times \text{Number of triangles}}{\text{Number of connected triples}}
$$
삼각형 :  세 정점이 서로 연결되어 있는 구조 (예 : A-B-C-A)

connected triple : 세 정점 중 하나가 중심이 되어 양쪽과 연결된 구조 (예 : A-B-C)

Clustering Coefficient 지표는 전체 그래프 안에 connected triple들이 실제로 삼각형을 이루는 비율을 말합니다.

즉, 그래프가 얼마나 잘 뭉쳐있는지를 나타내고, 2-star graph 중에 완벽한 삼각형의 비율을 의미합니다.

```R
> transitivity(karate, type = "global")
[1] 0.2556818
```



간단한 예제를 통해 이해해보겠습니다.

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L4\global_ex.jpg" alt="global_ex" style="zoom: 40%;" />
$$
\text{edge}: \{ (A-B), (A-C), (B-C), (B-D) \} \\
\text{Number of triangles : 1 (A-B-C connected)} \\
\text{Number of connected triples : 5} \\
\{ (A-B-C), (B-A-D), (B-A-C), (B-C-D), (C-A-B) \} \\
cl_T(G) = \frac{3 \times 1}{5}=0.6
$$


이 네트워크에서는 5개의 연결된 triple 중 1개만이 삼각형을 형성하고, 그 하나의 삼각형은 세 방향에서 볼 수 있어서 x 3을 해줍니다. 이는 전제적으로 약 60%의 연결된 triple이 삼각형으로 닫히게 됩니다.

```R
> edges <- c("A", "B",
+            "A", "C",
+            "B", "C",
+            "B", "D")
> 
> g <- graph(edges = edges, directed = FALSE)
> 
> transitivity(g, type = "global")
[1] 0.6
```





##### Local Clustering Coefficient

정점 v를 기준으로 d_v는 정점 v의 차수를 의미합니다. 즉, v와 연결된 이웃 수를 의미합니다.


$$
\tau_3(v) = \binom{d_v}{2}
$$
이건 v를 중심으로 만들어질 수 있는 connected triple의 수입니다. v의 이웃들 중에서 두 명을 뽑아 삼각형을 만들 수 있는 잠재적 경우의 수를 의미합니다.
$$
\tau_{\Delta}(v) : \text{number of triangles involving v}
$$
정점 v가 실제로 포함된 삼각형(triangle)의 수를 의미합니다.


$$
cl(v)= \frac{\tau_{\Delta}(v)}{\tau_3(v)}
$$
 분자는 실제 삼각형 수를 의미하고, 분모는 가능한 삼각형의 수를 의미합니다. 즉, 이웃 중 두 명이 연결될 수 있는 경우의 수를 의미합니다.

이 비율은 정점 v의 이웃들이 얼마나 서로 연결되어 있는지를 나타냅니다.



```R
> transitivity(karate, "local", vids = c(1, 34))
    Mr Hi    John A 
0.1500000 0.1102941 

> transitivity(karate, type = "global")
[1] 0.2556818
```

정점 1번 (Mr Hi) : 0.15, 정점 34번 (John A) : 0.11으로 이 값들은 해당 인물 주변의 친구들이 얼마나 서로 연결되어 있는지를 나타냅니다. 여기서 1번과 34번은 각각 리더지만, 이웃 노드들 간 연결이 매우 강하진 않습니다. 전체 네트워크의 global clustering coefficient가 약 0.256이었는데, 이 둘은 그보다 낮은 편입니다.



이를 간단한 예제로 이해해보겠습니다.

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L4\global_ex.jpg" alt="global_ex" style="zoom:40%;" />
$$
\text{edge}: \{ (A-B), (A-C), (B-C), (B-D) \}
$$


정점 B의 Local Clustering Coefficient를 계산해봅시다!

위의 그래프에서 B와 직접 연결된 정점들은 A,C,D입니다.
$$
d_B=3
$$

$$
\tau_3(B) = \binom{3}{2}=3 \\
\{(A-B-C),(A-B-D),(C-B-D)\}
$$

가 가능한 삼각형 후보인데, 



실제 삼각형 수는
$$
\tau_{\Delta}(B)=1
$$
가 됩니다.



이를 수식에 대입하면,
$$
cl(B)= \frac{\tau_{\Delta}(B)}{\tau_3(B)}=\frac{1}{3} \approx0.333
$$
이를 해석하면, 정점 B 주변에는 세 명의 이웃이 있고, 이 중 한 쌍만 서로 연결되어 있습니다.

따라서 B의 local clustering coefficient는 0.333 (33.%)라고 할 수 있습니다.



##### Reciprocity (Directed Graphs) 

Reciprocity에는 두 가지 정의를 할 수 있습니다.

Dyadic Reciprocity는 asymmetric dyads(일방향) 수 대비 mutual dyads(양방향) 수의 정도를 의미합니다.

Edge-based Reciprocity는 총 edges 수 대비 reciprocated dyads(양방향) 수의 정도를 의미합니다. 



계산을 해보면, 

asymmetric dyads(일방향) 수보다 총 edges 수의 개수가 더 많기 때문에(즉, Edge-based의 분모가 더 크기 때문에) Edge-based Reciprocity의 값이 더 작습니다.



```R
> reciprocity(aidsblog, mode = "default") # dyadic reciprocity
[1] 0.03278689
> reciprocity(aidsblog, mode = "ratio")  # edge-based reciprocity
[1] 0.01666667
```

AIDS blog의 네트워크 자료의 reciprocity 값을 확인해보니, 실제로 edge-based의 값이 더 작게 나타납니다.



또, AIDS blog 네트워크가 더 작은 reciprocity 값을 나타내는데, 이는 weakly connected하다고 볼 수 있습니다.



간단한 예제로 직접 계산해봅시다!!

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L4\recipro_dg.JPG" alt="recipro_dg" style="zoom:40%;" />

Dyadic Reciprocity를 먼저 계산해보면,
$$
\text{asymmetric dyads = 2, mutual dyads = 1} \rightarrow 1/2=0.5
$$


Edge-based Reciprocity를 계산해보면,
$$
\text{edges = 4, reciprocated edges=2} \rightarrow 2/4=0.5
$$


이 경우에는 Dyadic Reciprocity와 Edge-based Reciprocity 값이 동일합니다.



##### Connectedness and Components

만약, 그래프 내의 모든 정점 쌍에 대해 서로 도달할 수 있는 경로(path)가 존재하면, 그 그래프는 연결되어 있다(connected)고 합니다.

```R
> is_connected(yeast)
[1] FALSE
```

그래프가 완전히 연결되어 있지 않을 경우, 여러 개의 connected subgraph들이 생깁니다. 그 중 가장 큰 것을 giant component라고 부릅니다.

```R
> comps <- decompose(yeast)
> table(sapply(comps, vcount))

   2    3    4    5    6    7 2375 
  63   13    5    6    1    3    1 
```

yeast 그래프는 전체적으로 연결되어 있지 않습니다. 하지만, 대부분의 노드는 하나의 큰 component에 포함되어 있습니다. 이 giant component는 전체 2617개 노드 중 2375개를 포함합니다.
$$
\frac{2375}{2617} \approx 0.907 \rightarrow 90\%
$$


##### Small-world property

Small-world 네트워크는 Short average path lengths와 High clustering 두 가지 특성을 동시에 갖는 네트워크를 말합니다.

Short average path lengths는 네트워크 안의 임의 두 정점 사이의 평균 거리가 짧다는 의미이고, High clustering은 높은 군집 계수로 내 이웃들끼리도 서로 연결되어 있을 확률이 높다는 특징을 가집니다.

```R
> yeast.gc <- decompose(yeast)[[1]]
> mean_distance(yeast.gc)
[1] 5.09597
> diameter(yeast.gc)
[1] 15
> transitivity(yeast.gc)
[1] 0.4686663
```



##### Connectivity and Vulnerability

연결성(connectitivity)은 그래프가 얼마나 잘 연결되어 있는지, 또는 얼마나 쉽게 끊어질 수 있는지를 측정하는 지표입니다.

**Vertex connectivity (정점 연결성)**

그래프 G가 k-vertex connected하려면, 그래프에 있는 정점의 수가 k보다 많아야하고, 그 다음에는 아무 정점 k-1개를 제거하도 그래프가 여전히 끊어지지 않고 하나로 연결되어 있어야합니다. 최소한 몇 개의 정점을 제거해야 그래프가 분리될 수 있는가를 나타내는 수치입니다.

k의 값이 클수록 그래프는 정점 제거에 강한(덜 취약한) 구조라는 뜻입니다.



**Edge connectivity (간선 연결성)**

그래프 G가 k-edge connected라고 말하려면, 그래프에는 정점이 2개 이상 있어야 하고, 그 다음에는 아무 간선 k-1개를 제거해도 그래프가 계속 연결된 상태여야합니다. 최소한 몇 개의 간선을 끊어야 그래프가 분리되는가를 나타내는 값라고 할 수 있습니다. 

이 값도 클수록 네트워크가 간선 제거에 더 강하다는 것을 의미합니다.


$$
\text{Vertex connectivity ≤ Edge connectivity ≤ Minimum degree} \ d_{min}
$$
어떤 정점(vertex)을 제거하면 관련된 간선(edge)도 함께 사라지기 때문에, 정점 연결성은 항상 간선 연결성보다 같거나 작습니다.

Minimum degree는 그래프에 있는 모든 정점의 차수 중에서 가장 작은 값을 갖는 정점의 차수를 의미하는데, 만약 이 정점이 1개밖에 연결되어 있지 않다면, 그 정점을 제거하거나 연결된 간선을 끊으면 네트워크가 끊길 수 있어, 그래프 전체의 연결성이 그것 이상이 될 수 없습니다.



##### Cut Vertices and Articulation Points

Articulation Points(절단점)이란, 그래프에서 어떤 정점을 제거했을 때 그래프가 더 이상 하나로 연결된 상태가 아니게 된다면, 그 정점을 articulation point 또는 cut vertex라고 합니다. 즉, 그 정점을 지우면 네트워크가 끊어지는 중요한 연결 지점이 되는 노드입니다.

정점과 그 정점에 연결된 간선들까지 함께 제거했을 때, 그래프가 여러 조각으로 나뉘면 !! articulation point !!



```R
> yeast.cut.vertices <- articulation_points(yeast.gc)
> length(yeast.cut.vertices)
[1] 350
```



##### Menger's Theorem

그래프 이론에서 두 정점 사이의 연결성과 정보 흐름의 안정성을 동시에 설명해주는 중요한 정리로, 두 정점 사이의 독립적인 경로의 수와 해당 경로를 완전히 차단하기 위해 제거해야하는 정점 또는 간선의 최소 개수가 서로 정확히 같다는 것을 말한다.

유한한 무방향 그래프 G 내에서 서로 다른 정점 u와 v가 있을 때, 이 정리는 두 가지 형태로 주어집니다.

먼저, Vertex version입니다. 정점 u와 v 사이의 **모든 경로를 끊기 위해 제거해야 하는 최소 정점 수**는, u에서 v까지 존재하는 **서로 내부 정점을 공유하지 않는 경로들(vertex-disjoint paths)**의 최대 개수와 같습니다. 즉, "얼마나 쉽게 끊을 수 있는가"와 "얼마나 다양한 방식으로 연결되어 있는가"는 정확히 일치합니다.

두번째로, Edge version입니다. 정점 u와 v 사이의 모든 경로를 끊기 위해 제거해야하는 최소 간선 수는, u에서 v까지 존재하는 **서로 간선을 하나도 공유하지 않는 경로들(edge-disjoint paths)**의 최대 개수와 같습니다. 이때 각 경로는 간선만 공유하지 않으면 되고, 정점은 공유할 수 있습니다.

또, 이 정리는 그래프의 견고함, 즉 쉽게 끊어지지 않는 성질이 그래프 내에 존재하는 서로 독립적인 경로들의 수와 밀접하게 연관되어 있다는 사실을 말해줍니다. 다시 말해, 그래프가 얼마나 튼튼하냐는 그 안에 서로 다른 경로가 얼마나 풍부하게 존재하느냐에 달려 있습니다. 

만약 두 정점 사이에 존재하는 **서로 겹치지 않는 경로의 수 k**가 많을수록, 그만큼 해당 그래프는 정보 흐름이 더 안정적이고, **전체 연결이 쉽게 끊어지지 않는 구조**임을 의미합니다. 반대로, 이런 경로가 적다면, 소수의 정점이나 간선만 제거해도 두 정점 간의 연결을 쉽게 차단할 수 있게 됩니다. 이는 연결성이 낮은 그래프일수록 정보가 흐를 수 있는 경로가 제한적이며, 그 경로 중 하나만 제거되더라도 전체 흐름이 중단될 수 있다는 점에서 매우 중요합니다. 



##### Directed Graphs : Strong vs Weak Connectivity

Weak Connectivity는 방향을 무시하고 봤을 때, 그래프가 연결되어 있으면, "약하게 연결된 것"을 의미합니다. 즉, 방향을 무시하면 모든 노드 쌍이 하나의 component 안에 있습니다. 

```R
> is_connected(aidsblog, mode = "weak")
[1] TRUE
```



Strong Connectivity는 방향을 포함해서 모든 정점 쌍 u,v 사이에 u → v와 v → u 경로가 모두 존재하면 강하게 연결되었다고 합니다. 즉, directed graph 자체에서 모든 노드 쌍이 양방향 접근 가능해야합니다.

```R
> is_connected(aidsblog, mode = "strong")
[1] FALSE
```

AIDS blog 네트워크는 강하게 연결되지 않습니다.



Strongly Connected Component (SCC)

SCC는 그래프 내에서 서로 강하게 연결된 정점들의 집합입니다. 

aidsblog의 예시를 확인해보면, 다음과 같습니다.

```R
> aidsblog.scc <- components(aidsblog, mode = "strong")
> table(aidsblog.scc$csize)

  1   4 
142   1 
```

크기 1인 SCC가 142개 : 거의 모든 노드가 단독으로 고립되어 있습니다. 크기 4인 SCC가 1개 : 딱 4개의 정점만 서로 강하게 연결되어 있습니다.

AIDS blog 네트워크는 **약하게는 연결되어 있지만**, 진정한 **강한 연결은 극히 일부 (4개 정점)**에서만 발생합니다.

