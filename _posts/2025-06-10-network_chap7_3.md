---
published: true
layout: single
title:  "Network : Tomographic Network Topology Inference"
categories: Network
toc: true
use_math: true
---

이 포스팅은 서울시립대학교 통계학과 네트워크자료분석 수업 중 **'Network : Tomographic Network Topology Inference'**에 대한 내용을 담고 있습니다.

---

### Chapter 7  Network Topology Inference (Part 3)



#### Tomographic Network Topology Inference

##### Introduction

Tomographic Network Topology Inference란, 네트워크 그래프에서 내부의 정점(노드)와 엣지(선) 구조를 모두 관측할 수 없고 오직 일부 접근 가능한(외부, exterior) 노드에서만 측정이 가능한 경우 "내부 네트워크 구조 전체"를 추론하는 문제를 의미합니다. 

외부 노드(external nodes)는 우리가 측정 가능한 위치로 예를 들어, 네트워크의 끝, 관측소, 측정 장치가 연결된 지점이 있습니다. 내부 노드(internal nodes)는 직접적으로는 "측정 불가능"하거나 "숨겨진" 네트워크 내부의 지점을 의미합니다. 여기서 내가 데이터를 관측할 수 있는 곳이 외부가 됩니다.

이 추론은 큰 어려움을 가지고 있습니다. 관측 가능한 데이터만으로, 내부 전체 구조를 유일하게 복원하는 것은 거의 불가능에 가깝습니다. 왜냐하면, 같은 외부 관측값을 만들어내는 내부 구조가 여러 가지일 수 있기 때문인데요. 실제론, 내부 구조를 tree(트리)로 제한한다거나 다른 추가적인 모델에 대한 가정을 줘서, 문제가 너무 어려워지지 않게 "tractable"하게 만듭니다.

실제 예시로는 컴퓨터 네트워크가 있습니다. 네트워크 내부의 라우터나 노드는 직접 관측하지 못하고, 외부에서 패킷 지연, 도달 시간 등만 측정할 수 있습니다. 이는 내부 네트워크 구조를 추론할 수 있게 됩니다.



##### Constraining the Problem : Tree Topologies

Tree(트리)란, 사이클이 없고 모든 노드가 연결된 그래프입니다. 즉, "닫힌 순환(고리)"이 전혀 없고, "모든 정점이 한 덩어리로 이어진 구조"를 뜻합니다. 

Rooted Tree란, 뿌리(root)가 지정된 트리로 
$$
r \in V_T
$$
트리의 한 정점을 뿌리로 정합니다.

Leaves(잎노드, leaf nodes)는 degree가 1인 노드입니다. 즉, 끝에 붙어있는 노드로 

이들을 잎이라고 하고 보통은 R로 표기합니다.



전체 노드 집합에서 잎노드들과 뿌리노드를 뺀 나머지를 내부 노드라고 합니다. 


$$
V_T \setminus (R \cup \{ r \})
$$


즉, 트리 안쪽의 분기점 역할을 하는 노드들이 internal(내부) 노드라고 합니다.



트리의 엣지는 보통 branch(가지)라고 부릅니다. 

트리로 제한하면, 여러 가지 네트워크 구조 중 "가능한 경우의 수"가 확 줄어서 내부 구조 추정이 가능해집니다.

<img src="{{site.url}}\images\2025-06-10-network_chap7_3\tree.jpg" alt="tree" style="zoom:33%;" />



이 그림은 Rice University에서 다른 여러 기관으로 데이터를 전송하는 논리적 경로를 그린 예시입니다.

<img src="{{site.url}}\images\2025-06-10-network_chap7_3\riceECE.PNG" alt="riceECE" style="zoom:80%;" />

맨 위에 있는 Rice ECE 컴퓨터가 root(시작점, 출발 노드)입니다. 이 컴퓨터에서 데이터가 출발해서 네트워크를 타고 내려갑니다. 네트워크 라우터 아이콘(TX, IND 등)이 여러 단계에 걸쳐 나와있습니다. 이들은 실제로는 router나 중간 경유지로 외부에서 직접적으로 관측할 수 없는 숨겨진/중간 네트워크 노드입니다. 네트워크 데이터가 여러 방향으로 분기되는 분기점 역할을 합니다. 맨 아래쪽의 컴퓨터 아이콘들이 leaves입니다. 이는 실제 기관 또는 최종 컴퓨터에 해당됩니다. 관측이 가능한 외부 노드를 leaves 노드라고 합니다.(실제 데이터를 측정하거나 접근할 수 있는 지점을 뜻합니다.) 이 그림은 내부 네트워크 구조를 직접 볼 수 없고, 외부 노드에서만 측정 가능할 때, 트리 형태로 내부 구조를 추정하는 Tomographic Network Inference 예시입니다. 실제 라우터(내부 노드)는 숨겨져 있지만, 외부에서 데이터를 보낼 때 경로 상의 논리적 트리 구조를 그릴 수 있음을 보여줍니다.



**Tomographic Network Inference**에서 문제를 더 단순하게 만들기 위해 “**이진 트리(binary tree)**”로 구조를 제한하는 이유와, 실제로 **추론(inference)에서 목표가 무엇인지**를 설명합니다.

이진 트리(binary tree)는 모든 내부 노드가 자식이 최대 2개만 가질 수 있는 트리라고 합니다. 모든 트리(일반적인 트리)는 필요하다면 이진 트리로 변환해서 분석할 수 있습니다. 이렇게 하면 수학적으로 다루기 쉽고, 계산 복잡도가 줄어들어서 현실적으로 문제 해결이 쉬워집니다.

실제 추론 문제의 목표는 N_l개의 라벨이 붙은 leaf(관측 가능한 노드)가 있는 가능한 모든 이진 트리들의 공간에서 관측 데이터에 잘 맞는 트리를 찾는 것입니다. 이 데이터는 일반적으로 n번 독립적 측정에서 얻은 관측값들입니다. 선택적으로 각 엣지에 대한 가중치(edge weight)도 추정할 수 있습니다.

이진 트리로 구조를 제한하면 계산이 단순해지고, 관측 가능한 leaf가 주어졌을 때 측정 데이터에 가장 잘 들어맞는 트리(및 필요시 엣지 가중치)를 찾는 것이 tomographic inference의 목표다!

Tomographic network inference 문제는 다양한 실제 분야에서 중요한 역할을 합니다.
 예를 들어, **생물학의 계통발생학(phylogenetics)**에서는 여러 종의 유전자나 형태 데이터를 바탕으로 “진화 트리”를 복원할 때 이 문제가 등장하며, 이때 우리가 직접 관측할 수 있는 것은 현재의 종(leaf)뿐이고, 진화 과정에 존재했던 조상(내부 노드)은 직접 관측할 수 없습니다. 또한, **컴퓨터 네트워크 분야**에서도 인터넷의 내부 라우터나 경로 구조는 외부에서 직접적으로 알기 어렵지만, 패킷이 여러 경로를 거쳐 도달하는 최종 장치(leaf)에서 측정한 데이터를 이용해 숨겨진 내부 라우터와 경로(내부 노드)의 논리적 구조를 추론하게 됩니다. 따라서 tomographic network inference는 생물 진화 트리 복원과 네트워크 내부 구조 복원 등 다양한 분야에서 실질적으로 활용됩니다.



##### Example : Sandwich Probing

**네트워크 내부 구조를 추정하는 방법**으로 Sandwich Probing이라는 네트워크 계측 기법의 아이디어와 작동 원리를 설명하고 있습니다.

**Coates et al.**이 제안한 네트워크 내부(라우터, 경로 등) 구조 추정을 위한  **패킷 계측(probing) 방법**입니다.

**세 개의 probe packet(계측용 패킷)**을 순서대로 전송:

1. **작은 패킷**을 목적지 i로 전송 → delay1 (첫 도착 시간 측정)

2. **큰 패킷**을 목적지 j로 전송

3. 다시 **작은 패킷**을 목적지 i로 전송 → delay2 

   그리고 delay2 - delay1 (도착 시간 차이, 지연 차이) 측정



이 방법의 핵심 아이디어는 큰 패킷이 네트워크 내에서 큐잉(queuing)이나 처리 지연(processing delay)을 일으켜 그와 경로를 공유하는 노드들에서는 뒤따라오는 두 번째 작은 패킷의 도착 시간이 첫 번째보다 더 느려집니다. 즉, **두 작은 패킷 사이의 도착 시간 차이 (지연 차이, delay difference)**는 두 경로(i와 j)가 얼마나 많은 네트워크 구간을 공유하고 있는지 알려줍니다. 

이 delay difference 데이터를 모아서 네트워크 내부에서 경로가 어디서 겹치고, 어디서 분기되는지를 추정할 수 있습니다. 즉, 내부 트리 구조를 추정할 수 있게 됩니다. **sandwichprobe 데이터셋**에는 **delay difference 측정값**들이 담겨있습니다.



도착 시간 차이가 의미하는 것은 큰 패킷이 네트워크 중간 경로에서 큐잉(지연)이 발생할 때, 만약 i와 j가 네트워크에서 경로를 많이 공유하고 있다면 두번째 작은 패킷이 첫번째보다 더 많이 지연됩니다. 경로 공유가 적으면 delay 차이가 거의 없게 됩니다.

<img src="{{site.url}}\images\2025-06-10-network_chap7_3\delay.jpg" alt="delay" style="zoom:30%;" />



왼쪽의 트리 그림의 경우, root → 중간 branch → i, j로 같은 경로에서 나뉘는 구조 (즉, 경로 공유가 많음) 이고,

오른쪽 트리 그림의 경우, root에서 바로 i, j로 각각 다른 branch로 이어지는 구조로 경로 공유가 없습니다.



R코드를 통해 자세하게 살펴봅시다.

```R
> data(sandwichprobe)
> delaydata[1:5,]
  DelayDiff SmallPktDest BigPktDest
1       757            3         10
2       608            6          2
3       242            8          9
4        84            1          8
5      1000            7          3
> host.locs
 [1] "IST"    "IT"     "UCBkly"   "MSU1"   "MSU2"   "UIUC"  
 [7] "UW1"    "UW2"    "Rice1"    "Rice2"
```



DelayDiff : 두 작은 패킷의 도착 시간 차이 (sandwich probing에서 측정한 "delay difference")

여기서 leaf 노드는 총 10개입니다. (host.locs의 개수)





```R
> meanmat <- with(delaydata, by(DelayDiff, list(SmallPktDest, BigPktDest), mean))
> image(log(meanmat + t(meanmat)), xaxt="n", yaxt="n", col=heat.colors(16))
> mtext(side=1, text=host.locs, at=seq(0.0,1.0,0.11), las=3)
> mtext(side=2, text=host.locs, at=seq(0.0,1.0,0.11), las=1)
```

<img src="{{site.url}}\images\2025-06-10-network_chap7_3\heatmap.PNG" alt="heatmap" style="zoom:50%;" />

지연 시간 차이를 색상으로 heatmap에서 보여줍니다. 작은 지연 시간을 갖는 노드 i와 큰 지연 시간을 갖는 노드 j 의 차이를 heatmap으로 나타내고 있습니다.

Heatmap은 네트워크 내 노드들 사이의 pairwise delay differences (쌍별 지연시간 차이)를 색으로 표현한 것입니다.

빨간색이 진할수록 지연시간 차이가 적고 (더 가까운 관계), 노란색이 밝을수록 지연시간 차이가 크다는 뜻입니다.

IST, IT 노드 쪽은 노란색과 흰색이 섞여있어서 지연시간 차이가 크고, Rice1, Rice2 쪽은 진한 빨간색이라 지연시간 차이가 작습니다.

따라서 IST와 IT 사이가 매우 가깝고 (즉, 지연시간이 거의 없음), Rice1과 Rice2도 매우 가깝습니다.

UCBkly, MSU1, MSU2, UIUC 등도 어느 정도 중간 정도의 차이로 묶여있음을 알 수 있습니다.

그래서 heatmap만 보고 네트워크를 추정할 수 있습니다.



##### Tomographic Inference of Tree Topologies: An Illustration

두 가지 큰 방법의 분류가 존재합니다. 먼저, 거리 기반 군집화 방법으로 (i, j) 사이의 비슷한 애들끼리 묶습니다. 또, likelihood 기반 모델들인데, 우리는 거리 기반 군집화 방법에 집중합니다. 이는 tree 구조 형태가 대표적인데, 입력은 관찰된 데이터에서 유도된 불일치 행렬(dissimilarity matrix)로 됩니다.

arrival time이 낮다는 것은 두 개 사이에 공통된 path가 존재하지 않는다는 의미를 가집니다.



```R
> SSDelayDiff <- with(delaydata, by(DelayDiff^2, list(SmallPktDest, BigPktDest), sum))
> x <- as.dist(1 / sqrt(SSDelayDiff))
> myclust <- hclust(x, method="average")
> plot(myclust, labels=host.locs, axes=FALSE, ylab=NULL, ann=FALSE
```

여기서 sqrt(SSDelayDiff)는 time difference가 클수록 x값이 작아지게 되고, 만약 i와 j가 유사할 경우, 거리가 가깝다고 판정, i와 j가 유사하지 않을경우 멀다고 판정합니다. 유사도를 바탕으로 거리를 판정하는 과정입니다.



<img src="{{site.url}}\images\2025-06-10-network_chap7_3\treetoplo.PNG" alt="treetoplo" style="zoom:50%;" />



- **Rice1, Rice2**가 가장 가까운 root에 붙어 있음 → 네트워크 토폴로지에서 이 둘이 중심 근처에 위치함을 의미
- **IST, IT** 가 한 그룹으로 묶여 있음 (빨간색 원) → 경로(path)를 공유해서 지연시간 차이가 적다는 의미, 즉 path share를 잘 포착함
- **UCBkly, UW1, UW2**가 또 다른 그룹으로 묶임 (파란색 원) → 또 다른 의미 있는 클러스터를 형성함
- 필기에서 Rice1, Rice2 그룹은 'Good'이라고 표시 → 실제 네트워크 특성을 잘 반영한 클러스터링 결과임
- IST, IT도 path share capture 잘 됐다고 적혀 있음 → 두 노드가 네트워크 경로를 공유함을 잘 표현한 결과



- 이 그림에서 추론된 토폴로지는 실제(ground-truth) 네트워크의 특징을 많이 반영함

1. **Rice1과 Rice2는 root에 가깝다**
   - 두 노드가 네트워크의 중심에 위치함을 뜻함
2. **IST와 IT는 그룹으로 묶임 (Portugal 지역)**
   - 지역적, 네트워크적으로 가까운 노드들임
3. **MSU1, MSU2, UIUC는 Midwest 그룹 형성**
   - 미국 중서부 지역 노드들이 비슷하게 묶임
4. **Berkeley는 Wisconsin과 그룹이 됨 (Deviation)**
   - 이 부분은 실제 네트워크와 약간 다름 (예상과 다름)



위의 Rice University 네트워크를 확인해보면, 



<img src="{{site.url}}\images\2025-06-10-network_chap7_3\riceECE.PNG" alt="riceECE" style="zoom:80%;" />





올바르게 capture된 부분도 있고 그렇지 않은 부분도 있습니다. 이는 모델이나 데이터 한계, 측정 노이즈 때문이라고 추정합니다.

