---
published: true
layout: single
title:  "Network : Manipulating Network Data"
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Manipulating Network Data'**에 대한 내용을 담고 있습니다.

---





### Chapter 2 Manipulating Network Data

#### Manipulating Network Data

##### Network Representation

네트워크는 요소와 그 관계를 수학적으로 표현한 것입니다. 네트워크 데이터는 실생활에서 다양하게 사용됩니다. 대표적으로, 소셜 네트워크(Facebook, Twitter), 생물학적 네트워크(단백질 상호작용 네트워크), 교통 네트워크(항공사 노선망), 금융 네트워크(은행 간 대출 네트워크)에서 사용됩니다. 



#### Creating Network Graphs

##### Undirected Graph (무방향 그래프)



$$
G=(V,\ E) \\
u,v \in V , \ \{u,v\} \in E
$$



Vertices(점, set)는 Node라고도 합니다. Edges는 link라고 합니다.

$$
N_v= |V| : \text{order of the graph} \\
N_e= |E| : \text{size of the graph}
$$



**Graph Representation in R**

igraph 패키지에서는 igraph 클래스를 사용하여 그래프를 표현합니다. 여러 방법으로 igraph 객체를 표현할 수 있습니다.

```R
> library(igraph)
> g <- graph_from_literal(1-2,1-3,2-3,2-4,3-5,4-5,4-6,4-7,5-6,6-7) 
> plot(g)

> V(g) # List of vertices(nodes)
+ 7/7 vertices, named, from 954bfc8:
[1] 1 2 3 4 5 6 7

> E(g) # List of edges(links)
+ 10/10 edges from 954bfc8 (vertex names):
 [1] 1--2 1--3 2--3 2--4 3--5 4--5
 [7] 4--6 4--7 5--6 6--7
```

R의 함수 V(g)는 vertices의 list를 추출합니다. 또한, E(g)는 edges의 list를 추출합니다.

```R
> print_all(g) # Summary of graph
IGRAPH 954bfc8 UN-- 7 10 -- 
+ attr: name (v/c)
+ edges (vertex names):
1 -- 2, 3
2 -- 1, 3, 4
3 -- 1, 2, 5
4 -- 2, 5, 6, 7
5 -- 3, 4, 6
6 -- 4, 5, 7
7 -- 4, 6
```

print_all 함수에서 UN은 undirected graphs를 의미합니다. 옆의 (7,10)의 경우 (Num of node, Num of edge)를 의미합니다.

그래프로 나타낸다면 다음과 같습니다.

<img src="{{site.url}}\images\2025-04-10-network_chap2\undirected.PNG" alt="undirected" style="zoom:50%;" />



##### Directed Graphs (일방적 그래프)

방향 그래프(Directed Graph, Digraph)는 특정 방향을 가진 간선들로 구성됩니다. 방향 간선은 순서쌍 (u, v)로 표현되며, u에서 v로 향하는 방향 연결을 의미합니다. 간선을 directed edges 또는 arc라고 부릅니다.

(u, v)라는 아크에서 u는 꼬리(tail), v는 머리(head)입니다. 여기서 (u, v)와 (v, u)는 서로 다른 간선입니다! 방향성이 중요합니다!! Directed graphs는 비대칭적인 관계를 모델링할 때 유용하게 사용됩니다.

```R
> dg <- graph_from_literal(A-+B, A-+C, B++C)
> V(dg)
+ 3/3 vertices, named, from 7227394:
[1] A B C
> E(dg) # 양방향이 있기 때문에 4개!
+ 4/4 edges from 7227394 (vertex names):
[1] A->B A->C B->C C->B
> print_all(dg)
IGRAPH 7227394 DN-- 3 4 -- 
+ attr: name (v/c)
+ edges from 7227394 (vertex names):
[1] A->B A->C B->C C->B
```

이 코드를 통해 총 3개의 node(A,B,C), 4개의 edge로 이루어진다는 것을 알 수 있습니다. 또, Directed graph는 DN이라고 표현합니다.



그래프로 나타낸다면 다음과 같습니다.

<img src="{{site.url}}\images\2025-04-10-network_chap2\directed.PNG" alt="directed" style="zoom:50%;" />

##### Adjacency Representation(인접행렬을 통해 그래프 표현하기)

그래프는 인접 행렬로 표현할 수 있으며, 보통 A라고 표기합니다.



$$
A_{ij}=1 \ \ \text{if an edge exists from i to j}
$$





```R
> as_adjacency_matrix(dg) _ directed graph
3 x 3 sparse Matrix of class "dgCMatrix"
  A B C
A . 1 1
B . . 1
C . 1 .
```

```R
> as_adjacency_matrix(g) _ undirected graph
7 x 7 sparse Matrix of class "dgCMatrix"
  1 2 3 4 5 6 7
1 . 1 1 . . . .
2 1 . 1 1 . . .
3 1 1 . . 1 . .
4 . 1 . . 1 1 1
5 . . 1 1 . 1 .
6 . . . 1 1 . 1
7 . . . 1 . 1 .
```

다양한 그래프 표현 방식에서 그래프를 생성하는 함수들이 있습니다.

graph_from_adj_list : 인접 리스트로부터 그래프 생성

graph_from_edge_list : 간선 리스트로부터 그래프 생성

graph_from_adjacency_matrix : 인접 행렬로부터 그래프 생성



그래프 객체 입출력 함수에는 read_graph, write_graph가 있습니다.



##### Operations on Graphs

그래프는 vertices들과 edge를 추가하거나 제거하면서 수정될 수 있습니다. 

`igraph` 패키지는 이러한 작업들을 효율적으로 수행할 수 있는 함수들을 제공합니다.



##### Subgraph Extraction

subgraph는 더 큰 그래프에서 정점과 간선의 일부분을 선택하여 구성한 그래프입니다. 이 그래프는 더 큰 그래프 내에서 특정 네트워크의 특정 구조를 집중적으로 분석할 수 있게 됩니다. 

`induced_subgraph()` 함수는 특정 정점 집합을 지정하여 해당 부분 그래프를 추출합니다.

**유도 부분 그래프(induced subgraph)** 는 원래 그래프에서 **선택된 정점들 사이에 존재하는 모든 간선**을 포함합니다.

```R
h <- induced_subgraph(g,1:5) 
# graph g에서 1:5만 추출
plot(h)
```

<img src="{{site.url}}\images\2025-04-10-network_chap2\plot(h).PNG" alt="plot(h)" style="zoom:50%;" />

그래프는 정점 집합 V와 간선 집합 E로 구성되기 때문에, 정점이나 간선을 추가하거나 제거하는 작업은 set operation으로 이해할 수 있습니다. 많은 집합 연산들은 그래프 연산(graphs operations)으로 확장되며, 이를 통해 그래프를 유연하게 조작할 수 있습니다. 

```R
h <- g - vertices(c(6,7))
plot(h)
h <- h + vertices(c(6,7))
plot(h)
h <- h + edges(c(4,6), c(4,7), c(5,6), c(6,7)) 
plot(h)
# graph는 set의 pair (vertices, edge)
```



```R
h <- h + edges(c(4,6), c(4,7), c(5,6), c(6,7)) 
```

이 코드는 그래프를 직관적으로 나타낼 수 있는 방법을 제시한 코드입니다.



##### Graph Union and Intersection

두 그래프의 합집합(union)은 두 그래프의 모든 정점과 간선을 포함합니다. 두 그래프의 교집합(intersection)은 공통으로 존재하는 정점과 간선만 포함합니다.

이를 R코드로 나타낸다면,

```R
> matrix(c("A","B","B","C"), ncol=2, byrow=T)
     [,1] [,2]
[1,] "A"  "B" 
[2,] "B"  "C" 
> # (V1,E1) = ({A,B,C}, {A->B,B->C})
> matrix(c("B","C","C","D"), ncol=2, byrow=T)
     [,1] [,2]
[1,] "B"  "C" 
[2,] "C"  "D" 
> # (V2,E2) = ({B,C,D}, {B->C,C->D})
```

```R
g1 <- graph_from_edgelist(matrix(c("A","B","B","C"), ncol=2, byrow=T))
g2 <- graph_from_edgelist(matrix(c("B","C","C","D"), ncol=2, byrow=T))
g_union <- union(g1,g2)
g_intersection <- intersection(g1,g2)
```

이를 그래프로 표현하면 다음과 같습니다.

<img src="{{site.url}}\images\2025-04-10-network_chap2\graph_union.PNG" alt="graph_union" style="zoom:50%;" />



#### Decorating Network Graphs

Network Graph는 정점, 간선, 그리고 그래프 자체에 속성을 부여하며 더 다양하게 표현할 수 있습니다. 속성(attribute)는 시각화, 분석, 모델링 등에 사용될 수 있는 추가 정보를 제공합니다.

##### Vertex Attributes(정점 속성)

Vertex Attributes는 그래프 내 노드(정점)에 대한 메타데이터를 저장합니다. 속성은 범주형(성별, 역할, 그룹 라벨)일 수도 있고, 수치형(측정값, 중심성 점수)일 수도 있습니다. 일반적으로 많이 사용되는 속성에는 레이블, 색상, 그룹 구분, 인구통계 정보(나이, 성별) 등이 있습니다.

정점 속성은 아래와 같이 할당합니다: `V(graph)$속성명`

```R
> V(dg)$name = c("Sam", "Mary", "Tom")
> V(dg)$gender <- c("M", "F", "M")
```



##### Edge Attributes(간선 속성)

간선 속성은 노드 쌍 간의 관계에 대한 정보를 저장합니다. 

간선 속성에는 가중치, 유형, 타임스탬프가 있습니다. 가중치(weights)는 상호작용의 강도, 중요도를 표현하고, 유형(types)는 관계 유형(친구, 협업, 경쟁 등)을 표현하고, timestamp는 상호작용이 발생한 시점에 대한 정보를 저장합니다. 

간선 속성은 아래와 같이 할당합니다: `E(graph)$속성명`

간선에 가중치가 있는 그래프는 가중치 그래프(weighted graph)라고 합니다. 가중치는 일반적으로 0과 1 사이로 정규화됩니다.

```R
> is.weighted(g)
[1] FALSE

> wg <- g
> E(wg)$weight = 1:10
> is.weighted(wg)
[1] TRUE
```



##### Graph-Level Attributes(그래프 수준 속성_vertices와 edges)

그래프 속성은 개별 vertices나 edges가 아니라, 그래프 전체에 적용되는 정보를 의미합니다. 이러한 속성은 그래프의 metadata로 사용됩니다. 예를 들어, 그래프의 이름나 네트워크의 유형과 같은 정보들로 담을 수 있습니다.

그래프에 속성을 지정하는 방법은 두 가지 방법이 있습니다.

먼저, $ 연산자를 사용하는 방법이 있습니다.

```R
g$name <- "Example_Network"
g$type <- "Undirected_Graph"
```

위와 같이 `g$속성명` 형식으로 그래프에 직접 속성을 할당할 수 있습니다.



두번째로는 graph_attr() 함수를 사용할 수 있습니다.

```R
graph_attr(g, "name") <- "Example_Network"
graph_attr(g, "type") <- "Undirected_Graph"
```

`graph_attr()` 함수는 속성을 설정하거나 가져오는 데 사용할 수 있으며, 보다 명시적인 방식입니다.



##### Visualizing Attributes

Group Attributes는 네트워크 시각화를 보다 풍부하고 직관적으로 만들어줍니다. 정점의 색상(color), 크기(size), 레이블(label) 등은 각 정점의 속성 값에 따라 조정할 수 있습니다.

```R
install.packages("viridis")
library(viridis)

V(wg)$color <- viridis(length(V(g)))
plot(wg, vertex.color=V(wg)$color, edge.width=E(wg)$weight)
```



##### Using Data Frames to Define Graphs

데이터프레임(data frame) 은 네트워크 데이터를 저장하는 데 자주 사용되는 구조로, 특히 관계형 데이터셋에 적합합니다.

`graph_from_data_frame()` 함수는 간선 정보가 포함된 데이터프레임과 (선택적으로) 정점 속성 정보를 이용해 그래프를 생성합니다.

일반적으로 간선 리스트(edge list)는 `from`과 `to` 열로 구성됩니다.

```R
g.lazega <- graph_from_data_frame(
  elist.lazega,
  directed = FALSE,
  vertices = v.attr.lazega
)

g.lazega$name <- "Lazega Lawyers"
```



이 예제에서는,

`elist.lazega` 는 **간선 정보**를 담고 있고,

`v.attr.lazega` 는 **정점(노드)들의 속성 정보**를 담고 있습니다.



간선 데이터 (elist.lazega) :

```R
head(elist.lazega)
  V1  V2
1 V1 V17
2 V2  V7
3 V2 V16
4 V2 V17
5 V2 V22
6 V2 V26
```



정점 속성 데이터 (v.attr.lazega) :

```R
head(v.attr.lazega)
  Name Seniority Status Gender Office Years Age Practice School
1   V1         1      1      1      1    31  64        1      1
2   V2         2      1      1      1    32  62        2      1
3   V3         3      1      1      2    13  67        1      1
4   V4         4      1      1      1    31  59        2      3
5   V5         5      1      1      2    31  59        1      2
6   V6         6      1      1      2    29  55        1      1
```

여기서 `Seniority`, `Status`, `Gender`, `Office` 는 각 정점의 속성입니다.

이러한 정보를 기반으로 정점의 색상, 크기 등을 시각화에 활용할 수 있습니다.



##### Retrieving Graph Information

정점 속성(vertex attributes)은 데이터 프레임 형태로 포함될 수 있습니다. 각 행은 하나의 정점에 해당하고, 각 열은 속성 값을 나타냅니다. `graph_from_data_frame()` 함수의 `vertices` 인자를 사용하면, 정점 속성을 그래프에 통합할 수 있습니다.



```R
library(sand)

# 정점의 수 (vertices)
vcount(g.lazega)
[1] 36

# 간선의 수 (edge)
ecount(g.lazega)
[1] 115

# 그래프에 포함된 vertices 속성 리스트 확인
list.vertex.attributes(g.lazega)
[1] "name"      "Seniority" "Status"    "Gender"    
[5] "Office"    "Years"     "Age"       "Practice"  
[9] "School"
```



##### Visualizing Graphs Constructed from Data Frames

데이터프레임으로부터 그래프를 생성하면, 그에 포함된 vertices와 edge의 속성들을 이용하여 시각화할 수 있습니다. plot() 함수는 저장된 속성들을 활용해 그래프를 사용자 정의할 수 있도록 해줍니다.





#### More on Graphs

##### Basic Graph Concepts



$$
G=(V,\ E) \\
u,v \in V , \ \{u,v\} \in E
$$



그래프는 vertex들의 집합 V와 이 정점들을 연결하는 edge들의 집합 E로 구성됩니다.

그래프는 방향 그래프(directed graph) 또는 무방향 그래프(undirected graph)로 나뉠 수 있습니다. edge에 숫자값(가중치)이 포함되어 있다면, 가중치 그래프(weighted graph)라고 합니다. 



##### Adjacent and Neighboring Vertices

u와 v, **두 vertices가 edge(u,v)로 연결**되어 있다면, 이 두 vertices는 인접(adjacent)하다고 합니다. 어떤 정점의 이웃(**neighbors**)은 그 정점과 직접 연결된 다른 정점들입니다. 

<img src="{{site.url}}\images\2025-04-10-network_chap2\g.PNG" alt="g" style="zoom:50%;" />

```R
> neighbors(g,1) 
+ 2/7 vertices, named, from 1ca1fdb:
[1] 2 3
```

1번 노드와 직접 연결되어 있는 노드는 2번과 3번 노드입니다.

neighbor : 노드와 연결된 다른 노드 (각자의 노드)

어떤 edge e가 vertices v와 연결되어 있다면, 그 edge는 정점 v에 incident하다고 합니다. 즉, incident 간선이란, 특정 정점과 직접 연결된 모든 간선들입니다.

```R
> incident(g,2)
+ 3/10 edges from 1ca1fdb (vertex names):
[1] 1--2 2--3 2--4
```

이 결과는 정점 2에 연결된 간선이 1--2, 2--3, 2--4 세 개임을 보여줍니다.



##### Loops and Multigraphs

loop(루프)는 하나의 정점이 자기 자신과 연결된 간선(edge)입니다. 즉, 간선의 양 끝점이 동일합니다. multigraph(다중그래프)는 두 정점 사이의 여러 개의 간선(edge)이 존재하는 그래프입니다. 

loop나 multigraph는 교통망, 논문 인용 네트워크, 생물학적 시스템 같은 분야에서 자주 등장합니다.

루프나 다중 간선이 하나라도 포함된 그래프를 다중그래프(multigraph) 라고 부르며,  루프와 중복 간선이 없는 그래프를 단순 그래프(simple graph) 라고 합니다. 이때의 간선은 올바른 간선(proper edge) 라고도 불립니다.

```R
> g <- graph_from_literal(1-2,1-3,2-3,2-4,3-5,4-5,4-6,4-7,5-6,6-7) 
> plot(g)
> is_simple(g)  
[1] TRUE # simple graph
```

현재의 그래프 `g`는 루프나 중복 간선이 없어서 **단순 그래프**입니다.



```R
> mg <- g + edge(2,3) 
> print_all(mg)
IGRAPH d2b2c9e UN-- 7 11 -- 
+ attr: name (v/c)
+ edges (vertex names):
1 -- 2, 3
2 -- 1, 3, 3, 4
3 -- 1, 2, 2, 5
4 -- 2, 5, 6, 7
5 -- 3, 4, 6
6 -- 4, 5, 7
7 -- 4, 6
> is_simple(mg)
[1] FALSE # multigraph
```

정점 2와 3 사이에 **중복 간선**을 추가하자, 이제는 **단순하지 않은 그래프**가 되었습니다.



```
> E(mg)$weight <- 1
> wg2 <- simplify(mg)
> is_simple(wg2)
[1] TRUE
> print_all(wg2)
IGRAPH 446f15f UNW- 7 10 -- 
+ attr: name (v/c), weight (e/n)
+ edges (vertex names):
1 -- 2, 3
2 -- 1, 3, 4
3 -- 1, 2, 5
4 -- 2, 5, 6, 7
5 -- 3, 4, 6
6 -- 4, 5, 7
7 -- 4, 6
> E(wg2)$weight
 [1] 1 1 2 1 1 1 1 1 1 1
```

`simplify()` 함수를 사용하면 **중복 간선이 제거**되고, 다시 **단순 그래프**로 변환됩니다.

가중치를 처리하는 방법은 중복 간선을 제거할 때, **중복된 간선들의 가중치를 합산**할 수 있습니다.

`simplify()` 함수를 사용해도 weight를 통해 multiple 표시가 가능합니다!



##### Degree of Vertex

정점의 차수는 그 정점에 연결된 간선의 개수를 의미합니다.

방향 그래프(directed graph)의 경우 :

In-degree (입력 차수): 해당 정점으로 들어오는 간선의 개수

Out-degree (출력 차수): 해당 정점으로 나가는 간선의 개수

```R
> degree(dg)
A B C 
2 3 3 
> # the number of edges
> 
> degree(dg, mode="in")
A B C 
0 2 2 
> # the number of incoming edges
> 
> degree(dg, mode="out")
A B C 
2 1 1 
> # the number of outgoing edges
```

`igraph` 패키지의 `degree()` 함수는 정점들의 차수를 계산해줍니다.

방향 무시(전체 차수) : A는 2개의 간선, B는 3개, C도 3개의 간선에 연결되어 있습니다.

**in-degree** (들어오는 간선): A에게는 들어오는 간선이 없고, B와 C에게는 2개씩 있습니다.

**out-degree** (나가는 간선): A는 2개를 보내고 있고, B와 C에게는 각각 1개씩 나가는 간선이 있습니다.



##### Walk, Circuits, and Acyclic Graphs

Walk(경로) : 각 연속된 정점(vertice) 쌍이 간선(edge)으로 연결된 정점들의 나열

Circuit (Cycle, 회로/순환) : 시작점과 끝점이 같은 닫힌 경로 ex) {5,6,7,4,5} , {5,6,4,5}

Directed Graph에서 walk는 edge의 방향을 따라야하며, 유효한 경로가 되기 위해서는 edge의 방향을 그대로 따라야합니다.

DAG (Directed Acyclic Graph) : 방향이 있는 그래프 중, 방향을 따라 어떤 회로도 형성되지 않는 그래프입니다. 즉, 간선 방향을 따라가면서 순환은 발생하지 않습니다.



##### Reachability in Graphs

정점 v는 정점 u로부터 도달 가능(reachable)하다고 할 수 있습니다. 방향 그래프(directed graph)에서는 reachable은 edge의 방향에 따라 달라집니다.

distance() 함수는 정점 간의 도달 거리(최단 경로 길이)를 계산해줍니다. 각각의 값은 하나의 정점에서 다른 정점까지의 최소 edge 수를 나타냅니다.

<img src="{{site.url}}\images\2025-04-10-network_chap2\g.PNG" alt="g" style="zoom:50%;" />

```R
> distances(g)
  1 2 3 4 5 6 7
1 0 1 1 2 2 3 3
2 1 0 1 1 2 2 2
3 1 1 0 2 1 2 3
4 2 1 2 0 1 1 1
5 2 2 1 1 0 1 2
6 3 2 2 1 1 0 1
7 3 2 3 1 2 1 0
```

예를 들어, 정점 1에서 정점 4까지의 거리는 2입니다. 두 개의 edge를 거쳐 도달할 수 있습니다.



##### Paths and Connectivity

경로(path)는 vertices을 연결하는 edge의 연속입니다. 두 정점 사이의 거리(distance)는 그 정점들을 연결하는 최단 경로(shortest path) 상의 간선 수를 의미합니다.  `shortest.paths()` 함수는 그래프 내 모든 정점 간의 최단 거리 또는 특정 정점들 간의 최단 거리를 계산합니다.

```R
> shortest.paths(g, v=1, to=5)
  5
1 2
```

위 출력은 정점 1에서 정점 5까지 최단거리가 2임을 나타냅니다. 즉, 두 개의 edge를 거쳐 정점 5에 도달할 수 있다는 의미힙니다.



##### Connected Components

그래프는 모든 정점 쌍 사이에 경로가 존재하면 connected graph (연결된 그래프)라고 합니다. connected component (연결 요소)는 서로 연결된 정점들로 이루어진 최대 부분 그래프를 의미합니다.

```R
> is_connected(g)
[1] TRUE
```

`is_connected()` 함수는 그래프 전체가 하나의 연결 요소인지 (즉, 완전히 연결되어 있는지) 확인합니다.



`components()` 함수는 그래프 내의 **각 연결 요소(component)** 를 식별해줍니다.

```R
> components(g)
$membership
1 2 3 4 5 6 7 
1 1 1 1 1 1 1 

$csize
[1] 7

$no
[1] 1
```

- `$membership`: 각 정점이 어떤 연결 요소에 속하는지 보여줍니다.
   (여기서는 모든 정점이 `1번` 연결 요소에 속해 있음)
- `$csize`: 각 연결 요소의 정점 수 (여기서는 크기 7인 연결 요소 하나)
- `$no`: 연결 요소의 총 개수 (`1`개)

즉, 이 예시 그래프는 **모든 정점이 서로 연결된 하나의 연결 요소**로 구성된 그래프입니다.



그래프에서 지름(diameter)은 모든 정점 쌍 사이의 **최단 경로 중 가장 긴 거리**를 의미합니다. 즉, 그래프에서 가장 멀리 떨어진 두 정점 사이의 최단 경로의 길이를 의미합니다.

`diameter(g)` 함수는 그래프 내 가장 먼 두 정점 사이의 최단 경로 거리를 계산합니다.

```R
> shortest.paths(g, v=V(g), to=V(g))
```

위의 코드는 모든 정점 쌍 사이의 최단 거리 행렬을 보여줍니다.

```R
> shortest.paths(g, v=V(g), to=V(g))
  1 2 3 4 5 6 7
1 0 1 1 2 2 3 3
2 1 0 1 1 2 2 2
3 1 1 0 2 1 2 3
4 2 1 2 0 1 1 1
5 2 2 1 1 0 1 2
6 3 2 2 1 1 0 1
7 3 2 3 1 2 1 0
```

```R
> diameter(g)
[1] 3
```

이 그래프는 지름이 3임을 나타냅니다.



##### Special Types of Graphs

특정 그래프들은 고유한 특성과 구조를 가지고 있어 다양한 분야에서 사용되고 있습니다. 

대표적인 그래프 유형은 complete graph(완전 그래프), ring graph(링 그래프), bipartite graph(이분 그래프)가 있습니다.

**d-regular graph(d-정규 그래프)**는 모든 정점이 정확히 d개의 이웃 정점을 가지는 그래프입니다. 즉, 모든 정점의 차수가 동일하게 d입니다. 만약, 4-정규 그래프일 경우, 각 정점이 정확히 4개의 간선을 가집니다.

**complete graph(완전 그래프)**는 모든 정점 쌍이 하나의 고유한 간선으로 연결되어 있는 그래프입니다.

n개의 정점을 가지는 complete graph는 총



$$
\frac{n(n-1)}{2}
$$



개의 간선을 가집니다. 



**ring graph(링 그래프)**는 정점들이 원형(ring) 구조로 연결되어 있으며, 

각 정점들은 정확히 두 개의 이웃을 가집니다.

이를 R코드로 나타내본다면, 다음과 같습니다.

```R
g.dregular <- sample_k_regular(no.of.nodes=7, k=4)
g.full <- make_full_graph(7)
g.ring <- make_ring(7)

plot(g.dregular)
# d-regular graph

plot(g.full) 
# complete graph, d-regular graph

plot(g.ring)
# ring graph, d-regular graph
```

![complete]({{site.url}}\images\2025-04-10-network_chap2\complete.PNG)



##### Tree and Forests

tree(트리)는 cycle이 없는 하나의 단순 경로로 이루어진 연결 그래프입니다. 이는 데이터 표현의 기본 구조로 계층적 분류, 네트워크 라우팅도 포함합니다.

rooted tree(루트 트리)는 하나의 정점이 루트로 지정되어 방향성과 계층 구조를 부여한 구조합니다.

root는 tree의 가장 위에 있는 정점으로 parent 노드가 존재하지 않습니다. parent는 어떤 노드로 가는 경로에서 바로 위 단계에 있는 노드를 의미합니다. 그 아래에 연결된 아래쪽 노드들은 children이라고 부릅니다.  children 노드가 없는 노드를 leaf node 또는 terminal node라고 합니다. ancestors는 root부터 특정 노드까지의 경로에 포함된 모든 상위 노드를 가리킵니다. descendants는 특정 노드로부터 시작하여 하위로 이어지는 모든 노드를 의미합니다. 트리의 높이(height)는 root에서 가장 멀리 떨어진 앞 노드까지의 최장 경로의 길이를 뜻합니다.

만약, 그래프가 연결되지 않았지만, 연결된 각 구성 요소가 tree(트리)인 경우, 전체 그래프를 forest라고 합니다.

root : 1번 노드 , parent : 2번과 3번 노드 , children : 4,5,6,7번 노드 , leaf/terminal : 4,5,6,7번 노드

height : 2

<img src="{{site.url}}\images\2025-04-10-network_chap2\tree.png" alt="tree" style="zoom:40%;" />



##### k-Star Graphs

k-Star Graph는 하나의 root vertex가 k개의 잎 노드들과 직접 연결되어 있는 트리의 한 형태입니다. 이러한 구조는 **허브-스포크(hub-and-spoke)** 네트워크 모델에서 흔히 나타나며, 예를 들어 **교통망**, **사회 네트워크**, **중앙 서버 구조** 등에 적용된다.

```R
> g.tree <- make_tree(7, children = 2, mode="undirected")
> g.star <- make_star(7, mode="undirected")

> is_tree(g.tree)
[1] TRUE

> is_tree(g)
[1] FALSE

> is_tree(g.star)
[1] TRUE
```

<img src="{{site.url}}\images\2025-04-10-network_chap2\kstar.PNG" alt="kstar" style="zoom: 67%;" />





##### DAG

DAG(Directed Acyclic Graph, 방향 비순환 그래프)는 tree의 개념을 일반화한 그래프 구조 중 하나입니다. DAG는 방향성이 있는 그래프이며, cycle이 존재하지 않습니다. 즉, 어느 정점에서 출발해서 여러 정점을 따라가더라도, 자기 자신으로 되돌아오는 경로는 존재하지 않게 됩니다.

트리와 달리 DAG는 반드시 하나의 루트나 특정 계층 구조를 따르지 않으며, **여러 개의 부모 정점을 가질 수도 있고**, **하위 정점이 여러 상위 정점으로부터 영향을 받을 수 있습니다.** 따라서 DAG는 **계층 구조**뿐만 아니라 **의존 관계, 작업 순서, 버전 기록 등** 복잡한 구조를 표현하는 데 적합합니다.

```R
> is_dag(dg)
[1] FALSE
```

![dag]({{site.url}}\images\2025-04-10-network_chap2\dag.jpg)



##### Bipartite Graphs

이분 그래프(Bipartite Graph)는 vertices들이 두 개의 서로소 집합(disjoint sets)으로 나뉘며, 같은 집합에 속한 정점들끼리는 연결되지 않는 그래프입니다. 즉, 모든 edge는 반드시 하나의 집합에서 다른 집합으로 이어지는 구조를 가집니다. 이러한 구조는 주로 **서로 다른 두 종류의 개체 간 관계**를 모델링할 때 사용됩니다. 예를 들어, **영화 추천 시스템**에서는 사용자와 영화 간의 관계를 이분 그래프로 표현할 수 있습니다. 한쪽 정점 집합에는 사용자들이, 다른 쪽에는 영화들이 있고, 간선은 사용자가 본 영화 또는 좋아한 영화를 나타냅니다.

<img src="{{site.url}}\images\2025-04-10-network_chap2\Bipartite Graphs.jpg" alt="Bipartite Graphs" style="zoom:50%;" />

##### Projection of a Bipartite Graph

이분 그래프에서 한 집합(예:배우, 논문 저자 등)을 기준으로, 공통 이웃을 공유하는 노드끼리 연결해 새로운 단일 모드 그래프를 만듭니다.

예를 들어, 배우-영화 이분 그래프 일 때, 배우끼리 같은 영화에 출연했다면 연결됩니다. 또, 유저-상품 이분 그래프의 상황에서 유저끼리 같은 상품을 구매했다면 연결됩니다.

```R
# 이분 그래프의 프로젝션
proj <- bipartite_projection(g.bip)

# 첫 번째 프로젝션 (배우끼리의 관계)
print_all(proj[[1]])

IGRAPH dd3b1e1 UNW- 3 2-
+ attr: name (v/c), weight (e/n)
+ edges from dd3b1e1 (vertex names):
[1] actor1--actor2 actor2--actor3
```

위의 코드를 통해 actor 1과 actor 2는 movie1,  actor 2과 actor 3는 movie2에 함께 출연했기 때문에 연결됩니다.

`weight`는 **공동 이웃의 수**를 나타냅니다. (ex. 두 배우가 3개의 영화를 함께 했다면 weight = 3)



```R
# 두 번째 프로젝션 (영화끼리의 관계)
print_all(proj[[2]])

IGRAPH dd3b27a UNW- 2 1-
+ attr: name (v/c), weight (e/n)
+ edge from dd3b27a (vertex names):
[1] movie1--movie2
```

actor 2가 movie 1과 movie 2, 두 영화 모두에 출연했기 때문에 연결됩니다.



