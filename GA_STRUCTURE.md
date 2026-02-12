# GA (Genetic Algorithm) 구조 설명

## 1. 문제 정의

### RCMPSP (Resource-Constrained Multi-Project Scheduling Problem)
- **목표**: 여러 프로젝트를 제한된 팀(리소스)으로 스케줄링
- **목적함수**: 모든 프로젝트의 Tardiness(지연 시간) 합 최소화
  - Tardiness = max(0, 완료시간 - 납기)

### 문제 구성 요소
```
Project (프로젝트)
├── id: 프로젝트 ID
├── activities: 작업 리스트
├── release_time: 프로젝트 시작 가능 시간
└── due_date: 납기

Activity (작업)
├── id: 작업 ID
├── project_id: 소속 프로젝트
├── duration: 작업 수행 기간
├── eligible_teams: 해당 작업 수행 가능한 팀 리스트
├── predecessors: 선행 작업 리스트 (precedence constraint)
└── mutually_exclusive: 동시 수행 불가 작업 리스트 (mutex constraint)
```

---

## 2. Solution Structure (Random Key 방식)

### 2.1 핵심 개념

Random Key 방식은 **간접 표현(indirect representation)**을 사용하는 진화 알고리즘 인코딩 방법입니다.

```python
class Solution:
    def __init__(self, pairs: List[Pair]):
        self.pairs = pairs                    # 모든 (activity, team) 페어
        self.random_keys = [random.random()   # 각 페어에 대응하는 0~1 사이 실수
                           for _ in pairs]    
        self.objective: float = float('inf')  # 목적함수 값 (작을수록 좋음)
        self.fitness: float = 0.0            # 적합도 (클수록 좋음)
        self.schedule: Dict[int, Tuple[int, int, int]] = {}  
        # activity_id -> (start_time, end_time, team_id)
```

### 2.2 Solution의 구성 요소

#### (1) Pairs: (Activity, Team) 조합
```python
@dataclass
class Pair:
    activity_id: int  # 어떤 작업을
    team_id: int      # 어떤 팀이 수행할지
```

**Pair 생성 규칙**:
- 각 Activity마다 eligible_teams에 있는 팀과만 페어 생성
- eligible_teams가 없으면 모든 팀과 페어 생성

**예시**:
```
Activity 1 (eligible_teams: [0, 1])    → Pair(1, 0), Pair(1, 1)
Activity 2 (eligible_teams: [1, 2])    → Pair(2, 1), Pair(2, 2)
Activity 3 (eligible_teams: 없음)      → Pair(3, 0), Pair(3, 1), Pair(3, 2)
```

#### (2) Random Keys: 우선순위 값
- 각 Pair마다 0~1 사이의 실수값 할당
- **Random Key가 작을수록 우선순위가 높음** (먼저 스케줄링 시도)
- 유전 연산의 대상이 되는 실제 유전자

**예시**:
```
Pair(1, 0) → random_key = 0.234  (3번째 우선순위)
Pair(1, 1) → random_key = 0.891  (5번째 우선순위)
Pair(2, 1) → random_key = 0.012  (1번째 우선순위)
Pair(2, 2) → random_key = 0.567  (4번째 우선순위)
Pair(3, 0) → random_key = 0.123  (2번째 우선순위)
```

### 2.3 Random Key 방식의 장점

1. **항상 실행 가능한 해(Feasible Solution)를 생성**
   - Random key는 단순한 우선순위일 뿐, 제약조건은 디코딩 시 처리
   - Crossover/Mutation 후에도 항상 유효한 해 보장

2. **연속형 유전 연산자 활용 가능**
   - Random key는 실수값 → 표준 crossover/mutation 적용 가능
   - 순열(permutation) 기반 방법보다 간단하고 효율적

3. **다중 팀 할당 표현 가능**
   - 같은 Activity가 여러 팀과 페어를 형성
   - 디코딩 과정에서 가장 먼저 스케줄 가능한 팀 선택

---

## 3. Decoding Procedure (디코딩 절차)

Decoding은 **Random Key Solution을 실제 Schedule로 변환**하는 과정입니다.

### 3.1 디코딩 모드 2가지

#### (1) Batch Mode (기본 모드)
```
1. 모든 페어를 random_key 순서로 정렬
2. 정렬된 순서대로 스케줄링 시도
3. 한 번 순회 완료 후, 다시 처음부터 순회
4. 더 이상 스케줄할 수 없을 때까지 반복
```

#### (2) Immediate Mode (적극적 모드)
```
1. 모든 페어를 random_key 순서로 정렬
2. 정렬된 순서대로 스케줄링 시도
3. Activity 하나라도 스케줄되면 즉시 처음부터 다시 시작
4. 더 이상 스케줄할 수 없을 때까지 반복
```

### 3.2 디코딩 알고리즘 (Batch Mode 예시)

```python
def _decode_batch(self, solution: Solution):
    schedule = {}  # activity_id -> (start_time, end_time, team_id)
    team_available_time = [0] * num_teams  # 각 팀의 다음 가용 시간
    scheduled_activities = set()  # 이미 스케줄된 activity ID들
    
    # 페어들을 random_key 순서로 정렬 (한 번만)
    sorted_indices = sorted(range(len(pairs)), 
                          key=lambda i: solution.random_keys[i])
    
    # 더 이상 스케줄할 수 없을 때까지 반복
    while True:
        newly_scheduled = 0  # 이번 반복에서 새로 스케줄된 수
        
        for idx in sorted_indices:
            pair = pairs[idx]
            activity_id = pair.activity_id
            team_id = pair.team_id
            
            # 이미 스케줄된 activity면 skip
            if activity_id in scheduled_activities:
                continue
            
            # ★ 제약조건 체크
            if not check_constraints(activity_id, team_id):
                continue  # 제약 위반 시 skip (다음 반복에서 다시 시도)
            
            # ★ 스케줄 가능 → 스케줄에 추가
            start_time = calculate_earliest_start(activity_id, team_id)
            end_time = start_time + activity.duration
            schedule[activity_id] = (start_time, end_time, team_id)
            scheduled_activities.add(activity_id)
            newly_scheduled += 1
            
            # 팀 가용 시간 업데이트
            team_available_time[team_id] = end_time
        
        # 이번 반복에서 새로 스케줄된 것이 없으면 종료
        if newly_scheduled == 0:
            break
    
    return schedule
```

### 3.3 제약조건 체크 순서

디코딩 시 각 (Activity, Team) 페어에 대해 다음 제약조건을 순차적으로 체크합니다:

#### ① Precedence Constraint (선행 작업 완료)
```python
# 모든 선행 작업이 스케줄되어야 함
all_predecessors_done = True
earliest_start = project.release_time

for pred_id in activity.predecessors:
    if pred_id not in scheduled_activities:
        all_predecessors_done = False  # 아직 스케줄 안 됨 → skip
        break
    else:
        # 선행 작업의 완료 시간 이후에만 시작 가능
        pred_end_time = schedule[pred_id][1]
        earliest_start = max(earliest_start, pred_end_time)

if not all_predecessors_done:
    continue  # 다음 반복에서 다시 시도
```

#### ② Mutex Constraint (동시 수행 불가)
```python
# 동시 수행 불가인 activity가 이미 스케줄되어 있으면
# 그 activity가 끝난 후에 시작해야 함
for mutex_id in activity.mutually_exclusive:
    if mutex_id in scheduled_activities:
        mutex_end_time = schedule[mutex_id][1]
        earliest_start = max(earliest_start, mutex_end_time)
```

#### ③ Resource (Team) Availability
```python
# 해당 팀의 가용 시간 이후에만 시작 가능
earliest_start = max(earliest_start, team_available_time[team_id])
```

#### ④ Project Release Time
```python
# 프로젝트 시작 가능 시간 이후에만 시작 가능
earliest_start = max(earliest_start, project.release_time)
```

### 3.4 디코딩 예시

**입력 Solution**:
```
Pair(A1, T0) → key=0.8
Pair(A1, T1) → key=0.3  ← 2번째
Pair(A2, T0) → key=0.1  ← 1번째 (선행: A1 필요)
Pair(A2, T1) → key=0.9
Pair(A3, T0) → key=0.5  ← 3번째
```

**디코딩 과정** (Batch Mode):

**[1차 순회]**
```
1. Pair(A2, T0) 시도 (key=0.1) → 선행 작업 A1 미완료 → skip
2. Pair(A1, T1) 시도 (key=0.3) → 스케줄 성공
   Schedule: A1 = (0, 5, T1), T1 available = 5
3. Pair(A3, T0) 시도 (key=0.5) → 스케줄 성공
   Schedule: A3 = (0, 3, T0), T0 available = 3
```

**[2차 순회]**
```
1. Pair(A2, T0) 재시도 (key=0.1) → 선행 작업 A1 완료! → 스케줄 성공
   Schedule: A2 = (5, 9, T0)  (A1 종료 후 + T0 가용)
2. Pair(A1, T1) → 이미 스케줄됨 → skip
3. Pair(A3, T0) → 이미 스케줄됨 → skip
```

**[3차 순회]**
```
더 이상 스케줄할 것이 없음 → 종료
```

**최종 Schedule**:
```
Activity A1: T1, Start=0,  End=5
Activity A2: T0, Start=5,  End=9   (A1 완료 대기)
Activity A3: T0, Start=0,  End=3
```

### 3.5 Batch vs Immediate 차이점

| 구분 | Batch Mode | Immediate Mode |
|------|-----------|----------------|
| 재시작 시점 | 모든 페어 순회 완료 후 | Activity 하나 스케줄 시 즉시 |
| 우선순위 반영 | 느림 (여러 번 순회 필요) | 빠름 (즉시 반영) |
| 계산량 | 상대적으로 적음 | 상대적으로 많음 |
| 품질 | 좋음 | 더 좋을 수 있음 (우선순위 엄격 적용) |

**Immediate Mode의 특징**:
- Random key가 작은 페어가 가능하자마자 즉시 스케줄됨
- 우선순위가 낮은 페어는 후순위로 밀림
- Batch보다 random key의 영향력이 더 큼

---

## 4. 유전 연산자 (Genetic Operators)

### 4.1 Crossover (교차)

**Uniform Crossover** 사용:
- 각 페어의 random key를 부모 중 하나에서 50% 확률로 선택

```python
def crossover(parent1, parent2):
    child1, child2 = Solution(pairs), Solution(pairs)
    
    for i in range(len(pairs)):
        if random.random() < 0.5:
            child1.random_keys[i] = parent1.random_keys[i]
            child2.random_keys[i] = parent2.random_keys[i]
        else:
            child1.random_keys[i] = parent2.random_keys[i]
            child2.random_keys[i] = parent1.random_keys[i]
    
    return child1, child2
```

**예시**:
```
Parent1: [0.2, 0.5, 0.8, 0.3]
Parent2: [0.7, 0.1, 0.4, 0.9]
Random:  [0.3, 0.7, 0.2, 0.6]  (<0.5이면 P1, ≥0.5이면 P2)

Child1:  [0.2, 0.1, 0.8, 0.9]  (P1, P2, P1, P2)
Child2:  [0.7, 0.5, 0.4, 0.3]  (P2, P1, P2, P1)
```

### 4.2 Mutation (변이)

**Random Replacement Mutation**:
- 전체 페어의 10%를 랜덤하게 선택하여 새로운 random key 할당

```python
def mutate(solution):
    num_mutations = max(1, int(len(pairs) * 0.1))  # 10% 변이
    
    for _ in range(num_mutations):
        idx = random.randint(0, len(pairs) - 1)
        solution.random_keys[idx] = random.random()  # 새로운 랜덤 키
```

**예시**:
```
Before: [0.2, 0.5, 0.8, 0.3, 0.6, 0.1, 0.9, 0.4, 0.7, 0.2]
Mutate: index 2, 5를 선택 (10개 중 2개 = 20% → 10%로 조정됨)
After:  [0.2, 0.5, 0.45, 0.3, 0.6, 0.88, 0.9, 0.4, 0.7, 0.2]
                    ↑                     ↑
```

---

## 5. 목적함수 (Objective Function)

### 5.1 목적: Tardiness 최소화

```
Objective = Σ (각 프로젝트의 Tardiness)

where Tardiness_i = max(0, 완료시간_i - 납기_i)
```

### 5.2 계산 절차

```python
def calculate_objective(solution):
    # 1. 디코딩하여 스케줄 생성
    schedule = decode_and_schedule(solution)
    
    # 2. 프로젝트별 완료 시간 계산
    project_completion = {}
    for activity_id, (start, end, team) in schedule.items():
        project_id = activity_to_project[activity_id]
        project_completion[project_id] = max(
            project_completion.get(project_id, 0), 
            end
        )
    
    # 3. 프로젝트별 Tardiness 계산 및 합산
    total_tardiness = 0
    for project_id, completion_time in project_completion.items():
        due_date = projects[project_id].due_date
        tardiness = max(0, completion_time - due_date)
        total_tardiness += tardiness
    
    return total_tardiness
```

### 5.3 Penalty for Infeasible Solutions

스케줄되지 않은 activity가 있는 경우 (제약조건 위반):
```python
if len(unscheduled_activities) > 0:
    objective = len(unscheduled_activities) * 100000
```

**이유**: 
- 모든 activity를 스케줄하는 것이 최우선
- 불완전한 스케줄에 큰 벌점을 부여하여 진화 방향 유도

---

## 6. 선택 (Selection)

### Roulette Wheel Selection (룰렛 휠 선택)

**Fitness 변환**:
```python
# Objective → Fitness 변환 (클수록 좋게)
max_obj = max(solution.objective for solution in population)
epsilon = 1.0

for solution in population:
    solution.fitness = max_obj - solution.objective + epsilon
```

**선택 과정**:
```python
def roulette_wheel_selection(population):
    total_fitness = sum(sol.fitness for sol in population)
    pick = random.uniform(0, total_fitness)
    
    current = 0
    for solution in population:
        current += solution.fitness
        if current >= pick:
            return solution
```

**시각화**:
```
Population (fitness):
Sol1: fitness=50  [====================]
Sol2: fitness=30  [============]
Sol3: fitness=15  [======]
Sol4: fitness=5   [==]

Total fitness = 100
Pick random value in [0, 100]
→ Sol1 선택 확률: 50%
→ Sol2 선택 확률: 30%
→ Sol3 선택 확률: 15%
→ Sol4 선택 확률: 5%
```

---

## 7. 진화 알고리즘 흐름

```
1. 초기화: population_size개의 랜덤 solution 생성
   └─ 각 solution의 random_keys는 [0, 1] 사이 랜덤 값

2. 평가: 모든 solution 디코딩 + 목적함수 계산
   └─ objective 계산 → fitness 변환

3. 최적해 추적: 현재 population에서 최고 objective 기록

4. 진화 루프 (generations회 반복):
   
   a) Elitism: 최고 solution 1개 보존
   
   b) 새로운 population 생성:
      ┌─ Selection: Roulette Wheel로 parent1, parent2 선택
      ├─ Crossover: 확률 crossover_rate로 교차 수행
      │   └─ 수행 안 하면: 부모 그대로 복사
      └─ Mutation: 확률 mutation_rate로 변이 수행
      
   c) 평가: 새로운 population의 objective 계산
   
   d) 최적해 업데이트: 현재 best보다 좋으면 갱신

5. 종료: 최종 best solution 반환
```

---

## 8. 파라미터 설정

```python
GeneticAlgorithm(
    projects: List[Project],           # 프로젝트 리스트
    num_teams: int,                    # 팀(리소스) 개수
    population_size: int = 100,        # Population 크기
    generations: int = 500,            # 진화 세대 수
    crossover_rate: float = 0.8,       # 교차 확률 (80%)
    mutation_rate: float = 0.2,        # 변이 확률 (20%)
    tournament_size: int = 5,          # (사용 안 함, Roulette 사용)
    decode_mode: str = "batch"         # "batch" 또는 "immediate"
)
```

**권장 설정**:
- **Small instance** (activities < 50):
  - population_size=50, generations=200
  - decode_mode="immediate" (우선순위 엄격 적용)

- **Large instance** (activities > 100):
  - population_size=100, generations=500
  - decode_mode="batch" (계산 효율성)

---

## 9. Random Key 방식의 핵심 요약

### 9.1 인코딩
```
Solution = {pairs, random_keys}
         = {[(A1,T0), (A1,T1), (A2,T0), ...], [0.3, 0.8, 0.1, ...]}
         
"어떤 (Activity, Team) 페어를 먼저 시도할지"의 우선순위를 인코딩
```

### 9.2 디코딩
```
1. Random key 오름차순으로 페어 정렬
2. 정렬된 순서대로 스케줄링 시도
3. 제약조건 만족 시에만 스케줄 추가
4. 제약 위반 시 skip (다음 반복에서 재시도)
```

### 9.3 진화
```
- Crossover: Random key를 부모로부터 상속 (uniform)
- Mutation: Random key를 새로운 값으로 치환
- Selection: Fitness 기반 Roulette Wheel
```

### 9.4 장점
1. **항상 실행 가능한 해 생성**: 제약조건은 디코딩에서 처리
2. **간단한 유전 연산**: 실수값 배열에 표준 연산자 적용
3. **다중 옵션 표현**: 한 activity가 여러 팀과 페어 형성 가능
4. **우선순위 기반**: Random key가 스케줄링 순서를 자연스럽게 결정

---

## 10. 코드 구조

```
GA.py
├─ Activity         : 작업 정보 (duration, predecessors, mutex, eligible_teams)
├─ Project          : 프로젝트 정보 (activities, release_time, due_date)
├─ Pair             : (Activity, Team) 페어
├─ Solution         : Random Key 기반 solution
│   ├─ pairs        : 모든 (activity, team) 페어 리스트
│   ├─ random_keys  : 각 페어의 우선순위 (0~1 실수)
│   ├─ objective    : 목적함수 값
│   └─ schedule     : 디코딩된 실제 스케줄
│
└─ GeneticAlgorithm : GA 메인 클래스
    ├─ __init__                  : 문제 설정 및 페어 생성
    ├─ initialize_population     : 초기 population 생성
    │
    ├─ decode_and_schedule       : Solution → Schedule 변환
    │   ├─ _decode_batch         : Batch mode 디코딩
    │   └─ _decode_immediate     : Immediate mode 디코딩
    │
    ├─ calculate_objective       : Tardiness 계산
    ├─ evaluate_population       : Objective → Fitness 변환
    │
    ├─ roulette_wheel_selection  : Roulette 선택
    ├─ crossover                 : Uniform crossover
    ├─ mutate                    : Random replacement
    │
    ├─ evolve                    : 메인 진화 루프
    └─ print_solution            : 결과 출력
```

---

## 11. 사용 예시

```python
from GA import GeneticAlgorithm, Project, Activity

# 프로젝트 정의
activities1 = [
    Activity(id=1, project_id=1, duration=5, eligible_teams=[0,1], 
             predecessors=[]),
    Activity(id=2, project_id=1, duration=3, eligible_teams=[0,1], 
             predecessors=[1]),
]
project1 = Project(id=1, activities=activities1, release_time=0, due_date=10)

# GA 실행
ga = GeneticAlgorithm(
    projects=[project1],
    num_teams=2,
    population_size=50,
    generations=200,
    decode_mode="batch"
)

best_solution = ga.evolve()
ga.print_solution(best_solution)
```

---

## 12. 핵심 포인트

### ✅ Random Key의 역할
- **"스케줄링을 시도하는 순서"를 결정하는 우선순위**
- 제약조건은 디코딩 과정에서 자동으로 처리됨
- 유전 연산의 대상은 이 우선순위 값들뿐

### ✅ 디코딩의 역할
- Random key 순서대로 스케줄링 시도
- Precedence, Mutex, Resource 제약 모두 체크
- 제약 위반 시 skip하고 다음 반복에서 재시도

### ✅ 진화의 방향
- 목적함수: Tardiness 최소화
- 좋은 solution = Random key가 적절히 배치되어 제약조건을 빨리 만족하면서 
  납기를 지키는 스케줄을 생성하는 solution

### ✅ Batch vs Immediate
- **Batch**: 효율적이지만 우선순위 반영이 느림
- **Immediate**: 계산량이 많지만 random key 영향력이 큼
- 문제 특성에 따라 선택 (대부분 batch로 충분)
