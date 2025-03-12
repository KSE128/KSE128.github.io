---
published: true
layout: single
title:  "RDD(Resilient Distributed DataSet)"
categories: Economics
tag: Activity
toc: true
use_math: true
---

이 포스팅은 응용미시경제학 논문 작성 시 필요한 연구방법론 중 하나인 **RDD(Resilient Distributed DataSet)**에 대한 내용을 담고 있습니다.

---

논문 주소 : https://www.ewadirect.com/proceedings/ace/article/view/9896/pdf



### RDD (Resilient Distributed DataSet)

#### Introduction

RDD는 Apache Spark의 핵심 개념으로, 탄력적(Resilient), 분산(Distributed), 데이터셋(Dataset)의 속성을 가지고 있는 분산 데이터 처리 모델입니다. 기존의 **MapReduce, Storm** 등의 분산 데이터 처리 프레임워크에서는 병렬 처리 및 사용자 리소스 공유 문제 등이 존재했으며, **RDD는 이러한 문제를 해결하기 위한 핵심 데이터 모델**로 제안되었습니다. RDD는 **대량의 데이터를 여러 컴퓨터에서 동시에 처리할 수 있도록 설계된 분산 데이터셋**입니다.



#### Key Concepts of RDD

---

##### Meaning of RDD

- **Resilient (탄력적)**

  컴퓨터가 고장나더라도 데이터를 잃어버리지 않고 복구할 수 있습니다.

  Spark는 데이터 처리 과정을 기록해 두었다가, 문제가 생기면 다시 계산하여 복구합니다.

- **Distributed (분산된)**

  데이터가 한 대의 컴퓨터에 저장되지 않고, 여러 대의 컴퓨터에 나누어 저장됩니다.

- **Dataset (데이터셋)**

  RDD는 데이터를 저장하는 특별한 방식입니다.

  데이터는 리스트, 테이블, 텍스트 파일 등 다양한 형태로 존재하는데, Spark에서는 RDD형태로 변환하여 처리합니다.



##### Features of RDD

- **Horizontal Partitioning(수평적 분할)**

  RDD는 여러 개의 **파티션(Partition)**으로 나뉘며, 이는 **데이터 병렬 처리**를 가능하게 합니다.

- RDD의 처리 및 거부RDD의 분산 처리 및 파티셔닝**Elastic Storage(탄력적 저장)**

  메모리가 부족할 경우, RDD는 데이터를 디스크 또는 외부 저장소에 저장하여 **유연하게 처리**합니다.

- **Read-Only Dataset(읽기 전용 데이터셋)**

  한 번 생성된 RDD는 변경할 수 없으며, 기존 RDD에서 **새로운 RDD를 생성하는 방식**으로만 **데이터 변경**이 가능합니다.

- **Checkpointing(체크포인트)**

  반복 연산 중 RDD의 의존성이 커지는 것을 방지하기 위해, 특정 시점의 데이터를 **영구 저장소에 기록**할 수 있습니다.

- **Fault Tolerance(자동 장애 복구)**

  Spark는 **Lineage(연산 이력)**를 통해 데이터 손실 시 복구할 수 있도록 설계되어 있습니다.



#### Operations of RDD

---

##### Transformations 

변환 연산 : 기존 RDD에서 새로운 RDD를 생성하는 연산 (**Lazy Evaluation** 방식)

Lazy Evaluation(지연 평가)은 실제 연산을 즉시 수행하지 않고, 실행이 필요할 때까지 연산을 지연시키는 방식입니다.

**대표적인 변환 연산**

- `filter(func)`: 특정 조건을 만족하는 요소만 필터링

- `map(func)`: 각 요소에 함수를 적용하여 새로운 RDD 생성

- `flatMap(func)`: 여러 개의 출력 값을 생성하는 `map()`과 유사한 연산

- `groupByKey()`: `(Key, Value)` 형식의 데이터를 `Key` 기준으로 그룹화

- `reduceByKey(func)`: `Key`별 데이터를 `func`을 이용하여 집계



##### Actions 

행동 연산 : RDD에서 최종 결과를 반환하는 연산 (**즉시 실행**)

**대표적인 행동 연산**

- `count()`: 요소 개수 반환

- `collect()`: 전체 요소를 리스트로 반환

- `first()`: 첫 번째 요소 반환

- `take(n)`: 상위 `n`개 요소 반환

- `reduce(func)`: 두 개의 요소를 조합하여 하나의 값 반환
- `foreach(func)`: 각 요소에 대해 특정 함수를 실행



#### Distributed processing and partitioning of RDDs

---

RDD는 여러 개의 파티션으로 나뉘며, 각 파티션은 별도의 작업 노드에서 **병렬 연산**이 가능합니다.

Spark의 병렬 처리 성능을 최적화하려면 **파티션 수를 CPU 코어 수에 맞추는 것**이 중요합니다.

Spark는 기본적으로 **Hash Partition(해시 파티셔닝)**과 **Range Partition(범위 파티셔닝)**을 제공합니다.

`Spark.default.parallelism` 파라미터를 설정하여 기본 파티션 개수를 조정할 수 있습니다.



##### Hash Partition 

데이터를 해시 함수(Hash function)를 사용하여 특정 파티션에 배정하는 방식입니다. 같은 키를 가진 데이터는 항상 동일한 파티션으로 보내집니다. 키를 기준으로 그룹화(Groupby) 또는 조인(Join) 연산이 효율적입니다. 랜덤한 분배에 가까우며, 키의 개수가 많고 균등하게 분포되어 있을 때 효율적입니다. 그러나, 해시 충돌의 가능성이 있습니다. 일부 파티션에 데이터가 몰릴 수 있습니다. 또, 정렬이 필요할 경우 비효율적입니다.

대표적으로, `groupByKey()`, `reduceByKey()` 같은 키 기반 연산일 때 적합합니다.



##### Range Partition

데이터를 일정한 범위(Range) 단위로 나누어 파티션을 생성하는 방식입니다. 데이터의 정렬이 필요한 경우 적합하며, 값의 범위에 따라 정렬된 상태로 배분됩니다. 연속적인 키 값을 가진 데이터에서 성능이 우수합니다. 데이터가 정렬된 상태로 유지되어 범위 검색이 빠릅니다. 또, 비슷한 범위의 데이터가 같은 파티션에 위치하므로, 특정 범위 분석에 효과적입니다. 그러나, 데이터가 불균형하게 분포할 경우 특정 파티션에 데이터가 몰릴 가능성이 있습니다. 범위 기준을 미리 정해야 하므로, 데이터 분포를 미리 알고 있어야합니다.

대표적으로, 정렬 기반 분석, 범위 검색(`BETWEEN`, `WHERE a > b`)일 때 적합합니다.



#### RDD vs MapReduce

| 비교항목         | RDD (Spark)                       | MapReduce (Hadoop)              |
| :--------------- | :-------------------------------- | :------------------------------ |
| 데이터 저장      | 메모리 우선 (필요 시 디스크 저장) | 디스크 기반                     |
| 처리 속도        | 빠름 (메모리 중심)                | 느림 (디스크 Input/Output 많음) |
| 연산 방식        | Transformation + Action           | Map → Shuffle → Reduce          |
| 코드 단순성      | 간결한 API 제공                   | 복잡한 코드 구조                |
| 반복 연산 최적화 | Checkpoint, Caching 지원          | 반복 연산 시 비효율적           |

위의 비교를 통해 RDD는 MapReduce보다 훨씬 빠르고 효율적인 연산이 가능하며, 반복 연산이 필요한 머신러닝과 데이터 분석에서 강력한 성능을 발휘한다는 것을 알 수 있습니다.



#### Examples of experiments

---

To be continued



#### Conclusion

1. **RDD(Resilient Distributed Dataset)의 역할 및 성능 향상**

   기존 MapReduce 방식에서는 개별 Map 작업 결과가 다량으로 생성되고, Reduce 단계에서 데이터 전송량이 증가하여 성능 저하가 발생하게 됩니다. RDD는 **데이터의 불변성(Immutability)과 병렬 연산(Parallel Processing)** 을 활용하여 중간 결과 데이터를 효율적으로 관리하고, 네트워크 전송 부담을 줄여 전체적인 성능을 향상시킵니다. 이를 통해 MapReduce의 데이터 전송 실패율을 낮추고 실행 시간을 단축할 수 있습니다.

   

2. **빅데이터 분석과 RDD의 응용**

   RDD 기반의 Spark는 빅데이터 분석 시스템에서 중요한 역할을 하며, 클라우드 컴퓨팅과 결합하여 데이터 처리 속도를 높힙니다. 빅데이터는 전자상거래 추천 시스템, 금융 위험 예측, 물류 창고 최적화 등에 활용되며, 대량의 데이터 마이닝과 분석을 가능하게 합니다.

   

3. **Spark의 작업 스케줄링 문제와 해결 방안 (이해해야하는 부분)**

   Spark 클러스터의 이질성(heterogeneous cluster)으로 인해 작업 스케줄링의 비효율성이 발생할 수 있습니다. HSATS(Heterogeneous Spark Adaptive Task Scheduling) 전략을 적용하여, 사용자 및 아이템의 숨겨진 태그(hidden tag)를 벡터화하고, 유사도 계산을 통해 최적의 작업 스케줄링을 수행합니다.

   

4. **빅데이터 개발 환경과 RDD의 역할**

   효율적인 빅데이터 개발을 위해, 실시간 및 배치 처리 기능을 통합한 개발 환경이 필요합니다. 데이터 통합, 개발, 관리 등을 지원하는 IDE(통합 개발 환경)를 활용하면, SQL 작성만으로도 빅데이터 처리가 가능하도록 최적화할 수 있습니다. Spark RDD 기반의 데이터 처리 기술이 이러한 통합 플랫폼에서 핵심적인 역할을 할 수 있습니다.



