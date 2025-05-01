### Chapter 4 Descriptive Analysis of Network Graph Characteristics (Part 3)

#### Graph Partitioning

##### Concept and Motivation

Graph Partitioning이란, 그래프 G=(V,E)에서 vertex set V를 서로 겹치지 않는(cohesive, disjoint)  부분집합들로 나누는 과정입니다. 즉, 전체 정점들을 나누어 의미 있는 작은 집단들로 분할하는 과정입니다. 

이때 아래와 같은 조건을 만족해야합니다.
$$
C=\{C_1, ..., C_K\} \\
C_i \cap C_j = \emptyset \quad \text{for } \ i \ne j, \\
\bigcup_{k=1}^{K} C_k = V
$$
먼저, 서로 겹치지 않아야합니다.(disjoint 상태여야합니다.) 그리고, 모든 정점을 빠짐없이 포함해야합니다. 

좋은 Partition의 조건은 각 부분 집합은 내부적으로는 연결이 잘 되어 있어야 합니다. 또한, 다른 집합들과는 연결이 적어야합니다. (다른 커뮤니티와는 연결이 적어야합니다.) 이런 작업은 community detection라고도 불립니다. 복잡한 네트워크에서 서로 밀접하게 연결된 집단을 찾는데 사용됩니다.



##### Hierarchical Clustering for Graphs

Hierarchical Clustering (계층적 클러스터링)은 말 그대로 단계적으로 그룹을 만들어가는 방식입니다.

Agglomerative(병합형)과 Divisive(분할형), 두 가지 방식이 있습니다. 

Agglomerative(병합형)은 처음에는 각 정점을 각각 하나의 클러스터로 시작합니다. 그리고 가장 유사한 두 클러스터를 반복해서 병합합니다. 결과적으로 점점 큰 그룹들이 형성됩니다. (each vertex → group → merge)

Divisive(분할형)은 처음에는 모든 정점을 하나의 커다란 클러스터로 시작하는데, 내부에서 가장 나쁘게 연결된 부분을 찾아 분할하는 방식으로 점점 잘게 나눠갑니다.

여기서 핵심은 각 단계에서 좋은 병합 또는 분할을 선택하는 것입니다. 가장 널리 쓰이는 함수는 Modularity 방식입니다. 



##### Modularity

Modularity는 네트워크를 얼마나 잘 커뮤니티로 나누었는지 측정하는 함수입니다. 같은 커뮤니티 안에서는 간선이 많이 있고, 다른 커뮤니티 사이에는 간선이 적을수록 좋습니다. 이게 잘 드러나 있으면 높은 modularity를 가집니다.
$$
\text{Graph Partitioning} \ \ \ C= \{C_1, ..., C_K\}
\\

\text{mod}(C) = \sum_{k=1}^{K} \left[ f_{kk} - f_{kk}^* \right] \\
f_{kk}: \text{fraction of edges that have both endpoints in community} \ C_k \\
f_{kk}^*: \text{expected fraction of such edges under a null model}
$$
f_{kk} : 실제로 커뮤니티 내부에 존재하는 간선의 비율

f*_{kk} : 랜덤한 연결을 가정했을 때 기대되는 내부 연결 비율 → 즉, 우연히 그렇게 연결됐을 확률

위 둘의 차이가 크면 클수록, "랜덤으로는 설명 안되는 강한 내부 연결성"을 가지고 있다고 합니다. 즉, 진짜 커뮤니티가 있음을 의미합니다.
$$
\text{fraction matrix} \ f \ \text{is a} \ K  \times K \text{matrix such that} \\
[f]_{kk'} = \frac{ \sum_{i \in C_k} \sum_{j \in C_{k'}} A_{ij} }{ \sum_{i', j'} A_{i'j'} } \\
A_{ij} : \text{component of adjaceny matrix}
$$
행렬 f에서 분자는 vertex 집합 중 하나와 다른 vertex 집합 사이 실제 존재 간선 수를 의미하고, 분모는 전체 간선 수를 의미합니다. 

즉, f_{kk'}는 전체 간선 중에서 C_k와 C_k' 사이에 있는 간선의 비율을 의미합니다.



왜 f*_{kk}를 기대값이라고 하는가 !!
$$
f_{kk}^*=f_{k+} \cdot f_{+k}
$$
f_{k+} : C_k에 속한 노드가 전체 그래프의 얼마나 많은 간선을 차지하는지 의미합니다.

f_{+k} : 마찬가지로 C_k로 향하는 간선의 비율을 의미합니다.

즉, 우연히 C_k 내부에 연결이 생길 확률을 뜻합니다.

그래프에서 노드의 차수 분포 (degree distribution)는 유지하되, 간선은 무작위로 배치된다는 가정 하에 계산된 값입니다. 이걸 configuration model이리고 하며 이는 community 구조와 관계없이 연결이 무작위로 이루어지는 구성 모델에 해당합니다. 

|f_kk - f*_kk|의 높은 값은 예상보다 강력한 내부 응집력을 나타냅니다. 이는 의미있는 community 구조의 특징을 나타냅니다.



Modularity값을 가장 크게 만드는 분할을 직접 찾지 않는 이유는 가능한 분할의 수가 너무 많기 때문입니다. 이는 모두 계산하는 것이 불가능합니다. 

그래서 Greedy Heuristics를 사용합니다.



##### Greedy Search

Greedy Algorithm은 각 단계에서 가장 좋은 선택(locally optimal choice)을 반복해서 수행하는 방식입니다.

처음엔 각 정점이 자기 혼자 그룹을 가집니다. (agglomerative 시작) 그 다음에는 modularity가 가장 많이 증가하는 두 그룹을 병합합니다. 마지막으로 더 이상 개선이 없으면 종료됩니다.

이는 최적화된 modularity에 좋은 근사값을 제공합니다.

```R
> kc <- cluster_fast_greedy(karate)
> length(kc)
[1] 3
> sizes(kc)
Community sizes
 1  2  3 
18 11  5
```

kc는 fast greedy 알고리즘으로 찾은 커뮤니티 객체로 그 개수를 출력하면 3개가 됩니다. 각 커뮤니티의 크기를 출력하면 18,11,5가 나오는데, 가장 큰 그룹(18의 크기)은 John A이고, 두번째로 큰 그룹(11의 크기)은 Mr.Hi가 됩니다. 

```R
> plot(kc, karate)
```

그래프 상에서 각 커뮤니티가 다른 색으로 표시되어 한눈에 보기 좋습니다!



```R
> library(ape)
> dendPlot(kc, mode = "phylo")
```

dendrogram(덴드로그램)은 계층적 구조를 보여주는 나무 형태를 시각화한 것으로 높은 위치에서 병합될수록 두 클러스터 간의 유사도가 낮다는 의미를 지니고 있습니다.

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L5\hirarch.PNG" alt="hirarch" style="zoom: 50%;" />

가장 큰 그룹은 John A (admin,34)의 그룹에 해당하고, 두번째로 큰 그룹은 Mr. Hi(instr,1)의 그룹에 해당됩니다. 

덴드로그램에서는 클러스트링 과정을 tree 형태로 시각화한 것으로 클러스터 두 개가 합쳐지는 높이는 두 클러스터 간의 유사하지 않은 정도(= dissimilarity) 또는 병합할 때 드는 비용 (= cost)을 나타냅니다. 

덴드로그램에서 높이가 높은 병합일수록 두 클러스터는 서로 덜 비슷한 그룹이 합쳐졌다고 해석합니다.



##### Spectral Clustering for Graphs

Spectral Clustering은 그래프의 연결성을 행렬의 고유값과 고유벡터를 통해 분석하는 클러스터링 기법입니다. 그래프를 행렬로 표현하고 그 행렬의 고유 구조를 분석해서 자연스럽게 분리되는 부분을 찾아냅니다. 이 방식은 그래프의 연결성 속성을 그래프와 연관된 행렬의 고유 구조, 특히 그래프 라플라시안과 연관시킵니다.



**Graph Laplacian**

주어진 그래프 G에서 :
$$
L=D-A \\
A : \text{adjacency matrix} \\
D : \text{degree matrix}
$$
행렬 A는 정점 간의 연결 여부를 나타냅니다. (연결이 있으면 1, 없으면 0)

행렬 D는 정점의 차수 (연결된 간선 수)를 대각선에 놓은 행렬(diag)입니다.

즉, Laplacian은 연결 구조를 수학적으로 정리한 행렬입니다.



**Spectral Graph Theory의 핵심 결과**로

그래프가 K개의 연결 성분(connected components)을 가진다면, 다음의 결과가 나타납니다.
$$
\lambda_1 = \lambda_2 = \cdots = \lambda_K = 0,\quad \lambda_{K+1} > 0
$$
즉, **Laplacian L의 0인 고유값의 개수는 연결 성분의 개수**

→ 그래프의 구조를 고유값만으로 파악할 수 있습니다.

L은 대칭 행렬(symmetric)이고, positive semi-definite입니다. 가장 작은 고유값은 항상 0이고, 

이에 대응하는 고유벡터는 다음과 같습니다.
$$
1 = (1,1,...,1)^T \\
L \ \cdot\ 1=0
$$
이 벡터는 정점 간에 아무 차이도 없는 벡터입니다. 그래프 전체를 하나의 연결된 덩어리로 봤을 때, 모든 노드가 균등하게 연결되어 있으므로, 라플라시안은 이 벡터를 아무 변화 없이 그냥 0으로 보낸다는 뜻입니다. 즉, 람다는 0, 벡터는 1이라는 고유값과 고유벡터 쌍이 존재합니다.

고유값 0의 중복도(multiplicity)는 연결 성분의 개수를 나타내고, 고유값은 항상 0 이상으로 이루어져 있습니다.



Laplacian의 또 다른 표현이 존재합니다. (Incidence Matrix)
$$
B \in \mathbb{R}^{n \times m} : \text{oriented incidence matrix of the graph}
$$
각 행은 vertex(정점), 각 열은 edge(간선)을 의미합니다. 각 원소는 다음과 같이 정의됩니다.
$$
B_{i,e} =
\begin{cases}
+1 & \text{if vertex } i \text{ is the head of edge } e \\
-1 & \text{if vertex } i \text{ is the tail of edge } e \\
0  & \text{otherwise}
\end{cases}
$$


그러면, Laplacian 행렬은 다음과 같이 표현될 수 있습니다.
$$
L=BB^T = D-A
$$


**Graph Laplacian의 예시**


$$
\text{Vertices :} \ V= \{1,2,3,4\} \\
\text{Edges : } \ E=\{e_1=(1,2), e_2=(2,3),e_3=(3,4)\} \\
B \in \mathbb{R}^{4 \times 3} : \text{oriented incidence matrix} \\
B = \begin{bmatrix}
+1 & 0 & 0 \\
-1 & +1 & 0 \\
0 & -1 & +1 \\
0 & 0 & -1
\end{bmatrix}
$$

$$
L = B B^\top = 
\begin{bmatrix}
1 & -1 & 0 & 0 \\
-1 & 2 & -1 & 0 \\
0 & -1 & 2 & -1 \\
0 & 0 & -1 & 1
\end{bmatrix}
$$

$$
B B^\top \cdot \mathbf{1} = B (B^\top \cdot \mathbf{1}) = B \cdot \mathbf{0} = \mathbf{0}
$$

이것은 Laplacian 행렬이 엣지의 흐름(edge flows)이 정점의 불균형(vertex imbalances)으로 어떻게 전이(translate)되는지를 표현한다는 것을 보여줍니다.

그래프에서 엣지는 흐름의 경로이고, 그 흐름이 어디에서 쏠리고(유입), 어디에서 빠져나가는지(유출)는 정점(vertex) 단위로 나타납니다. Laplacian은 edge의 흐름을 계산해주는 장치입니다.



##### Laplacian의 물리적 의미 &  엣지 차이가 정점 간 불균형으로 연결되는 과정

"그래프 위에서의 **discrete divergence operator**"
$$
L = B B^\top
$$

$$
\text{edge flow vector : } \ f \in \mathbb{R}^{m} = \text{fixed orientation}
$$

여기서 **벡터 f**는 edge별로 그 흐름의 크기를 나타낸 벡터입니다.


$$
\text{The incidence matrix} \ B\in \mathbb{R}^{n \times m} \text{maps edge flows to vertices via :} \\
Bf \in \mathbb{R}^{n}
$$
즉, 결과 벡터 Bf는 각 정점에서의 순 유입량을 나타냅니다. (유입 - 유출)

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L5\result.PNG" alt="result" style="zoom: 70%;" />



(전압, 온도, 높이 등) 정점마다 값을 부여한다면, 

n차원에 있는 벡터 x는 정점 위에 정의된 스칼라 필드라고 생각합니다.
$$
Lx = BB^\top x = B(B^\top x)
$$

$$
B^\top x \in \mathbb{R}^{m} : \text{the differences of values across edges}
$$

이제는 반대로, 정점 간의 차이를 엣지로 보냅니다.
$$
\text{when, value of } e=(i,j) : \ x_i-x_j \ \  (\text{assuming the orientation is from} \ i \ \text{to} \ j)
$$

$$
B(B^\top x) = Lx : \text{vertex-level imbalance}
$$

edge-level 차이들을 다시 정점 기준으로 합산합니다. 그래서 **"각 정점이 주변 정점들과 얼마나 다른가?"**를 나타냅니다!



정리하면, 

- 1차로: \( B^\top x \) → **이웃과의 차이 계산** (local difference)  
- 2차로: \( B(B^\top x) \) → 그 차이들을 **정점마다 모아 불균형 계산** (vertex-level imbalance)



종합적으로 Laplacian은 local difference에서 global imbalance 변환기가 됩니다. 

Lx=0이라면, 각 정점이 이웃들과 균형 상태에 있다는 의미입니다. 

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L5\last!.PNG" alt="last!" style="zoom: 70%;" />



##### Spectral Clustering for Graphs: Fiedler Value and Fiedler Vector

Spectral Clustering에서 핵심적인 역할을 하는 Fiedler Value와 Fiedler Vector에 대한 내용입니다. 

L의 가장 작은 고유값은 항상 0이 존재합니다.
$$
\lambda_1=0 \rightarrow x_1=(1,...,1)^T
$$
고유값은 항상 0으로 존재하고 대응 고유벡터는 위의 x1 벡터와 같습니다. 이는 정점들이 모두 균일한 상태일 때의 벡터로 불균형이 존재하지 않습니다. 



만약 그래프가 두 덩어리로 분리될 수 있다면, 두번째로 작은 고유값(Fielder Value)은 0에 가까운 값이 됩니다. 
$$
\lambda_2 \approx0 \rightarrow x_2= \text{Fiedler Vector}
$$
 Fielder Value가 작다는건 그래프를 자연스럽게 잘라낼 수 있는 지점이 있다는 의미를 가집니다.

x2는 Fielder Vector는 정점마다 값을 주는데, 이 값을 기반으로 vertex를 두 그룹으로 나눌 수 있습니다.



Fiedler Vector를 이용하여 그래프를 분할하는 과정을 알아보면,

x2의 부호를 기준으로 vertex를 두 그룹으로 나눕니다.
$$
S = \{ v \in V : x_2(v) \ge 0 \}, \quad 
\bar{S} = \{ v \in V : x_2(v) < 0 \}
$$
즉, Fiedler vector의 각 성분을 확인하고, 0보다 크면 그룹 S, 0보다 작으면 그룹 S bar로 설정합니다. 

고유벡터는 Laplacian의 성질에 따라 정점 간 연결성과 차이를 반영합니다. 같은 쪽의 값일수록 비슷한 연결성을 가지고 있고, 부호를 기준으로 나누면 자연스러운 분리를 만들 수 있게 됩니다. 

edge cut를 최소화하여 S와 S bar 사이의 edge를 최대한 적게 하면서, 내부 연결은 많게 하려는 목적을 가지고 있습니다.



이 과정을 R코드를 통해 Karate 데이터를 활용하여 알아보겠습니다.

```R
> k.lap <- laplacian_matrix(karate) 
> eig.anal <- eigen(k.lap)
```

`laplacian_matrix(karate)`: karate 그래프의 **Laplacian 행렬 L=D-A** 을 생성

`eigen(k.lap)`: 이 Laplacian 행렬의 **고유값과 고유벡터** 계산



```R
> plot(eig.anal$values, col="blue", ylab="Eigenvalues of Graph Laplacian")
```

Laplacian의 **고유값들을 시각화**

작은 고유값이 왼쪽에, 큰 고유값이 오른쪽에 그려짐

**λ₁ = 0**부터 시작하고, **λ₂ (Fiedler value)**가 중요!



```R
> f.vec <- eig.anal$vectors[,33]
```

x2 :  **두 번째로 작은 고유값에 해당하는 고유벡터**

고유벡터는 `eigen()` 함수의 결과에서 **33번째 열**이 Fiedler vector입니다. 

(igraph에서는 정점 수 34 → 고유값도 34개. vcount(karate)를 이용해서 알 수 있음!

가장 작은 고유값이 첫 번째니까 두 번째 작은 값이 마지막에서 두 번째인 33번째 열에 있음)



```R
> faction <- vertex_attr(karate, "Faction")
> f.colors <- as.character(length(faction))
> f.colors[faction == 1] <- "red"
> f.colors[faction == 2] <- "cyan"
```

**정점 속성 `Faction`** (John A 그룹 or Mr. Hi 그룹) 가져옴

시각화를 위해 그룹 1은 **red**, 그룹 2는 **cyan**으로 표시



```R
> plot(f.vec, pch=16, xlab="Actor Number", ylab="Fiedler Vector Entry", col=f.colors)
> abline(0, 0, lwd=2, col="lightgray")
```

x축: 정점 번호 (1~34)

y축: Fiedler vector 각 성분 값

색깔: 소속된 faction (빨강 vs 시안)

`abline(0, 0)` → 기준선 (0) 추가해서 부호 나누는 기준 시각화

<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L5\eigenvalue.PNG" alt="eigenvalue" style="zoom:50%;" />

여기서 L의 고유값 중 정확히 0인 것이 하나뿐이라는 건, 이 그래프가 완전히 연결되어 있다는 듯입니다.

만약, 연결 성분(component)이 K개면, 0인 고유값도 K개가 나옵니다. 즉, 0이 하나라는 것은 하나의 연결된 그래프임을 알 수 있습니다.

두번째로 작은 고유값이 작다는 것은 그래프를 두 부분으로 거의 나눌 수 있다는 의미를 가지며, 이 그래프는 bipartite (이분 그래프) 처럼 행동한다는 뜻입니다. 특히, 클러스터 간 연결은 적고 내부 연결은 많다는 구조를 암시합니다.

Fiedler vector의 부호를 기준으로 나눴더니, 실제 karate club에 존재했던 두 파벌(John A와 Mr. Hi)의 분리와 정확히 일치합니다. 이는 spectral clustering이 실제 사회적 구조를 정확히 포착해냈다는 증거가 됩니다.



##### Spectral Clustering for Graphs: Generalizations and Extensions

두 개 이상의 커뮤니티가 필요할 때는 Fiedler vector를 이용한 분할을 재귀적으로 반복합니다. 

위에서 언급한 spectral clustering은 Fiedler vector의 두 부호로 두 그룹으로 나누는 과정을 의미합니다. 만약, 3개 이상으로 나누고 싶으면, 나눈 후에 각 그룹에 다시 spectral clustering을 적용합니다. 한 번에 K개로 나누는 게 아니라, 계속 두 개씩 쪼개는 방식으로 K개를 만든다고 생각하면 됩니다. 

네트워크 과학자 **Mark Newman**은 Laplacian 대신 **modularity matrix B** 를 사용하는 유사한 방법을 제안했습니다. Laplacian은 연결성 기반의 수행하는 반면, Modularity matrix는 커뮤니티 구조를 직접적으로 최적화하려는 행렬입니다. 즉, 내부 연결은 많고, 외부 연결은 적은 구조를 찾는 메커니즘입니다.

R의 igraph 패키지에서 Newman's 방법은 다음과 같습니다.

```R
> cluster_leading_eigen(graph)
```



#### Assortativity and Mixing

##### Assortativity and Mixing

Assortativity(동류성)은 네트워크에서 비슷한 특성을 가진 vertex끼리 서로 연결되는 경향을 가지고 있습니다. 같은 그룹끼리 연결이 많을 경우, assortative라고 하고, 다른 그룹끼리 연결이 많을 경우, disassortative라고 합니다. 이를 Assortative mixing이라고 합니다.



##### Assortativity and Mixing : Assortativity Coefficient (Categorical)

네트워크의 정점들이 M개의 유형으로 분류(범주형 특성)되어 있을 때, 

f_{ij} : type-i 정점과 type-j 정점 사이를 잇는 간선의 비율 : 전체 edge에서 i ↔ j로 연결된 간선이 얼마나 되는지


$$
f_{i+} = \sum_j f_{ij}, \quad
f_{+j} = \sum_i f_{ij}
$$
f_{i+} : type-i 정점에서 출발하는 간선의 전체 비율

f_{+j} : type-j 정점으로 들어오는 간선의 전체 비율

→ 이때 undirected graph라면 f_{i+} =  f_{+i}



**Assortativity Coefficient 수식**
$$
r_a = 
\frac{ \sum_i f_{ii} - \sum_i f_{i+} f_{+i} }
     { 1 - \sum_i f_{i+} f_{+i} }
$$
<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L5\assortativity_coef.PNG" alt="assortativity_coef" style="zoom: 67%;" />
$$
r_a\approx1 \ \text{: perfect assortative mixing} \\
r_a\approx0 \ \text{: mixing is random(null model)} \\
r_a\approx -1 \ \text{: strong disassortative mixing} \\
$$


간단한 예시를 통해 이해해보겠습니다!

학교 내에서 동아리 유형별로 다음과 같이 edge 비율이 있다고 해봅시다.

|      | A    | B    |
| ---- | ---- | ---- |
| A    | 0.4  | 0.1  |
| B    | 0.1  | 0.4  |

$$
\sum_i f_{ii}=f_{AA}+f_{BB}=0.4+0.4=0.8 \\
f_{A+}=0.5=f_{+A} \\
f_{B+}=0.5=f_{+B} \\
r_a = \frac{ 0.8 - \{(0.5\cdot0.5)+(0.5\cdot0.5)\} }
     { 1 - \{(0.5\cdot0.5)+(0.5\cdot0.5)\} } = \frac{0.8-0.5}{1-0.5}=\frac{0.3}{0.5}=0.6
$$

위의 예시는 undirected graph의 예시입니다. 

또, assortativity가 0.6으로 나타나 꽤 강한 assortativity를 가지고 있습니다.



이번 예시는 directed graph입니다.

| From \ To | A    | B    | C    |
| --------- | ---- | ---- | ---- |
| A         | 1    | 2    | 1    |
| B         | 0    | 1    | 1    |
| C         | 1    | 2    | 1    |

$$
f_{A+}= \frac{4}{10}=0.4 , \ \ f_{B+}= \frac{2}{10}=0.2 ,\ \ f_{C+}= \frac{4}{10}=0.4  \\
f_{+A}= \frac{2}{10}=0.2 , \ \ f_{+B}= \frac{5}{10}=0.5 ,\ \ f_{+C}= \frac{3}{10}=0.3 \\
\sum_i f_{ii}=f_{AA}+f_{BB}+f_{CC}=0.1+0.1+0.1=0.3 \\
r_a = 
\frac{ 0.3 - \{(0.4\cdot0.2)+(0.2\cdot0.5)+(0.4\cdot0.3)\} }
     { 1 - \{(0.4\cdot0.2)+(0.2\cdot0.5)+(0.4\cdot0.3)\} }= \frac{0}{0.7} \approx 0
$$

0의 값이 나타나 무작위 연결과 비슷하다는 의미를 가지고 있습니다.



##### Assortativity and Mixing : added 

Yeast example

네트워크 :  yeast protein interaction network 

목표 : 단백질 분류가 'P'인 노드들끼리 얼마나 서로 잘 연결되는지 확인

```R
> assortativity_nominal(yeast, (V(yeast)$Class=="P")+1, directed=FALSE)
[1] 0.4965229
```

약 0.5의 assortativity 계수를 가집니다. 이는 꽤 강한 assortative mixing이 있다는 의미를 가지며, class P는 서로끼리 연결되는 경향이 큽니다.

`assortativity_nominal()` 함수는 범주형 변수에 대한 **커뮤니티 유사 연결 경향**을 수치로 보여줍니다.



##### Assortativity and Mixing : Assortativity Coefficient (Numerical)

vertex의 특성이 numerical(수치형)일 때, 비슷한 노드끼리 연결되려는 성향을 분석합니다.
$$
r = \frac{\sum_{x, y} x y \left( f_{xy} - f_{x+} f_{+y} \right)}{\sigma_x \sigma_y} \\
f_{xy} : \text{fraction of edges connecting values x and y} \\
f_{x+},f_{+y} : \text{marginal sums over rows/columns}
$$
<img src="C:\Users\김충남\Desktop\2025-1\네트워크자료분석\L5\pearson.PNG" alt="pearson" style="zoom:67%;" />

이것은 피어슨 상관계수 (Pearson correlataion coefficient)의 일반적 형태입니다.

상관계수가 1에 가깝다면, 값이 비슷한 vertex끼리 강하게 연결되어있고, -1에 가깝다면 값이 서로 다른 vertex끼리 주로 연결됩니다. 또, 0이라면 무작위 연결과 유사합니다.

```R
> assortativity_degree(yeast)
[1] 0.4610798
```

가장 많이 쓰이는 특수한 경우는 degree assortativity로 yeast 단백질 상호작용 네트워크에서 degree와 degree 간 연결 상관관계를 구합니다. 상관계수를 확인해보면, 양수이므로 degree가 높은 단백질은 다른 고도 연결된 단백질과 연결되는 경향이 있습니다. biological network에서 흔히 나타나는 특징입니다.
