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



### RDD (Resilient Distributed DataSet, 탄력적 분산 데이터셋)

#### Introduction

---

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

- **Elastic Storage(탄력적 저장)**

  메모리가 부족할 경우, RDD는 데이터를 디스크 또는 외부 저장소에 저장하여 **유연하게 처리**합니다.

- **Read-Only Dataset(읽기 전용 데이터셋)**

  한 번 생성된 RDD는 변경할 수 없으며, 기존 RDD에서 **새로운 RDD를 생성하는 방식**으로만 **데이터 변경**이 가능합니다.

- **Checkpointing(체크포인트)**

  반복 연산 중 RDD의 의존성이 커지는 것을 방지하기 위해, 특정 시점의 데이터를 **영구 저장소에 기록**할 수 있습니다.

- **Fault Tolerance(자동 장애 복구)**

  Spark는 **Lineage(연산 이력)**를 통해 데이터 손실 시 복구할 수 있도록 설계되어 있습니다.



#### Operations of RDD

---

##### Transformations (변환 연산)

기존 RDD에서 새로운 RDD를 생성하는 연산 (**Lazy Evaluation** 방식)

Lazy Evaluation(지연 평가)은 실제 연산을 즉시 수행하지 않고, 실행이 필요할 때까지 연산을 지연시키는 방식입니다.

**대표적인 변환 연산**

- `filter(func)`: 특정 조건을 만족하는 요소만 필터링

- `map(func)`: 각 요소에 함수를 적용하여 새로운 RDD 생성

- `flatMap(func)`: 여러 개의 출력 값을 생성하는 `map()`과 유사한 연산

- `groupByKey()`: `(Key, Value)` 형식의 데이터를 `Key` 기준으로 그룹화

- `reduceByKey(func)`: `Key`별 데이터를 `func`을 이용하여 집계



##### Actions (행동 연산)

RDD에서 최종 결과를 반환하는 연산 (**즉시 실행**)

**대표적인 행동 연산**

- `count()`: 요소 개수 반환

- `collect()`: 전체 요소를 리스트로 반환

- `first()`: 첫 번째 요소 반환

- `take(n)`: 상위 `n`개 요소 반환

- `reduce(func)`: 두 개의 요소를 조합하여 하나의 값 반환
- `foreach(func)`: 각 요소에 대해 특정 함수를 실행
