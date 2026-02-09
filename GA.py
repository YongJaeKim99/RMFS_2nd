"""
RCMPSP (Resource-Constrained Multi-Project Scheduling Problem) 유전 알고리즘 (GA)
Random Key 방식:
- (Activity, Team) 페어를 생성
- 각 페어에 0~1 사이의 랜덤 키 할당
- 랜덤 키가 작은 순서대로 스케줄링 시도
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Activity:
    """Activity 정보"""
    id: int
    project_id: int
    duration: int
    eligible_teams: List[int]  # 이 activity를 수행할 수 있는 팀 리스트
    predecessors: List[int]  # 선행 activity ID 리스트
    mutually_exclusive: List[int] = None  # 동시 수행 불가 activity ID 리스트
    
    def __post_init__(self):
        if self.mutually_exclusive is None:
            self.mutually_exclusive = []


@dataclass
class Project:
    """프로젝트 정보"""
    id: int
    activities: List[Activity]
    release_time: int  # 프로젝트 시작 가능 시간
    due_date: int  # 납기


@dataclass
class Pair:
    """(Activity, Team) 페어"""
    activity_id: int
    team_id: int
    
    def __hash__(self):
        return hash((self.activity_id, self.team_id))
    
    def __eq__(self, other):
        return self.activity_id == other.activity_id and self.team_id == other.team_id
    
    def __repr__(self):
        return f"(A{self.activity_id}-T{self.team_id})"


class Solution:
    """
    Solution Structure (Random Key 방식):
    - pairs: 모든 (activity, team) 페어의 리스트
    - random_keys: 각 페어에 대응하는 0~1 사이의 실수값
    - 디코딩 시 random_key가 작은 순서대로 스케줄링 시도
    - objective: 목적함수 값 (작을수록 좋음)
    - fitness: 선택을 위한 적합도 값 (클수록 좋음)
    """
    
    def __init__(self, pairs: List[Pair]):
        self.pairs = pairs  # 모든 (activity, team) 페어
        self.random_keys = [random.random() for _ in pairs]  # 각 페어의 랜덤 키
        self.objective: float = float('inf')  # 목적함수 값 (작을수록 좋음)
        self.fitness: float = 0.0  # 적합도 값 (클수록 좋음)
        self.schedule: Dict[int, Tuple[int, int, int]] = {}  # activity_id -> (start_time, end_time, team_id)
        
    def copy(self):
        """Solution 복사"""
        new_sol = Solution(self.pairs)
        new_sol.random_keys = self.random_keys.copy()
        new_sol.objective = self.objective
        new_sol.fitness = self.fitness
        new_sol.schedule = self.schedule.copy()
        return new_sol


class GeneticAlgorithm:
    """유전 알고리즘 메인 클래스 (Random Key 방식)"""
    
    def __init__(
        self,
        projects: List[Project],
        num_teams: int,
        population_size: int = 100,
        generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        tournament_size: int = 5,
        decode_mode: str = "batch"  # "batch" 또는 "immediate"
    ):
        self.projects = projects
        self.num_teams = num_teams
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.decode_mode = decode_mode  # "batch" 또는 "immediate"
        
        # 모든 activities를 수집
        self.activities: List[Activity] = []
        self.activity_dict: Dict[int, Activity] = {}
        for project in projects:
            self.activities.extend(project.activities)
            for act in project.activities:
                self.activity_dict[act.id] = act
        
        # Project별 매핑
        self.project_dict = {p.id: p for p in projects}
        self.activity_to_project = {act.id: act.project_id for act in self.activities}
        
        # 모든 (activity, team) 페어 생성
        self.pairs: List[Pair] = []
        for activity in self.activities:
            if activity.eligible_teams:
                # eligible한 팀만 페어 생성
                for team_id in activity.eligible_teams:
                    self.pairs.append(Pair(activity.id, team_id))
            else:
                # 모든 팀과 페어 생성
                for team_id in range(num_teams):
                    self.pairs.append(Pair(activity.id, team_id))
        
        print(f"총 {len(self.pairs)}개의 (Activity, Team) 페어 생성됨")
        
        # 최적 solution 추적
        self.best_solution: Solution = None
        self.best_fitness_history: List[float] = []
        
    def initialize_population(self) -> List[Solution]:
        """초기 population 생성 - Random Key 방식"""
        population = []
        
        for _ in range(self.population_size):
            # 각 페어에 대해 0~1 사이의 랜덤 키를 할당
            solution = Solution(self.pairs)
            # random_keys는 이미 Solution __init__에서 초기화됨
            population.append(solution)
        
        return population
    
    def decode_and_schedule(self, solution: Solution) -> Dict[int, Tuple[int, int, int]]:
        """
        Random Key 방식으로 Solution을 실제 스케줄로 변환
        
        Mode:
        - "batch": 모든 페어를 한 번 순회한 후 다시 처음부터 (기존 방식)
        - "immediate": activity가 스케줄되면 즉시 처음 페어부터 다시 시작
        
        Returns: activity_id -> (start_time, end_time, team_id)
        """
        if self.decode_mode == "batch":
            return self._decode_batch(solution)
        elif self.decode_mode == "immediate":
            return self._decode_immediate(solution)
        else:
            raise ValueError(f"Unknown decode_mode: {self.decode_mode}")
    
    def _decode_batch(self, solution: Solution) -> Dict[int, Tuple[int, int, int]]:
        """
        Batch Mode (기존 방식):
        모든 페어를 한 번 순회한 후 다시 처음부터
        """
        schedule = {}  # activity_id -> (start_time, end_time, team_id)
        team_available_time = [0] * self.num_teams  # 각 팀이 사용 가능한 시간
        scheduled_activities = set()  # 이미 스케줄된 activity ID들
        
        # 페어들을 random_key 순서로 정렬 (한 번만)
        sorted_indices = sorted(range(len(solution.pairs)), 
                              key=lambda i: solution.random_keys[i])
        
        # 더 이상 스케줄할 수 없을 때까지 반복
        max_iterations = len(self.activities)  # 최대 반복 횟수 제한
        iteration = 0
        
        while iteration < max_iterations:
            newly_scheduled = 0  # 이번 반복에서 새로 스케줄된 activity 수
            iteration += 1
            
            # 정렬된 순서대로 스케줄링 시도
            for idx in sorted_indices:
                pair = solution.pairs[idx]
                act_id = pair.activity_id
                team_id = pair.team_id
                
                # 이미 이 activity가 스케줄되었으면 skip
                if act_id in scheduled_activities:
                    continue
                
                activity = self.activity_dict[act_id]
                project = self.project_dict[activity.project_id]
                
                # 선행 작업 완료 체크
                all_predecessors_done = True
                earliest_start = project.release_time
                
                for pred_id in activity.predecessors:
                    if pred_id not in scheduled_activities:
                        # 선행 작업이 아직 스케줄되지 않음
                        all_predecessors_done = False
                        break
                    else:
                        # 선행 작업의 완료 시간 고려
                        pred_end_time = schedule[pred_id][1]
                        earliest_start = max(earliest_start, pred_end_time)
                
                # 선행 작업이 완료되지 않았으면 skip (다음 반복에서 다시 시도)
                if not all_predecessors_done:
                    continue
                
                # 동시 수행 불가 제약 확인 (mutually exclusive)
                for mutex_id in activity.mutually_exclusive:
                    if mutex_id in scheduled_activities:
                        # 동시 수행 불가인 activity가 이미 스케줄되어 있으면
                        # 그 activity가 끝난 후에 시작해야 함
                        mutex_end_time = schedule[mutex_id][1]
                        earliest_start = max(earliest_start, mutex_end_time)
                
                # 팀의 가용 시간 확인
                earliest_start = max(earliest_start, team_available_time[team_id])
                
                # 스케줄에 추가
                start_time = earliest_start
                end_time = start_time + activity.duration
                schedule[act_id] = (start_time, end_time, team_id)
                scheduled_activities.add(act_id)
                newly_scheduled += 1
                
                # 팀 가용 시간 업데이트
                team_available_time[team_id] = end_time
            
            # 이번 반복에서 새로 스케줄된 activity가 없으면 종료
            if newly_scheduled == 0:
                break
        
        return schedule
    
    def _decode_immediate(self, solution: Solution) -> Dict[int, Tuple[int, int, int]]:
        """
        Immediate Mode (새로운 방식):
        activity가 하나라도 스케줄되면 즉시 처음 페어부터 다시 시작
        """
        schedule = {}  # activity_id -> (start_time, end_time, team_id)
        team_available_time = [0] * self.num_teams  # 각 팀이 사용 가능한 시간
        scheduled_activities = set()  # 이미 스케줄된 activity ID들
        
        # 페어들을 random_key 순서로 정렬 (한 번만)
        sorted_indices = sorted(range(len(solution.pairs)), 
                              key=lambda i: solution.random_keys[i])
        
        # 더 이상 스케줄할 수 없을 때까지 반복
        max_iterations = len(self.activities) * len(solution.pairs)  # 최대 반복 횟수 (더 많이)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            scheduled_in_this_pass = False  # 이번 pass에서 뭔가 스케줄되었는지
            
            # 정렬된 순서대로 스케줄링 시도
            for idx in sorted_indices:
                pair = solution.pairs[idx]
                act_id = pair.activity_id
                team_id = pair.team_id
                
                # 이미 이 activity가 스케줄되었으면 skip
                if act_id in scheduled_activities:
                    continue
                
                activity = self.activity_dict[act_id]
                project = self.project_dict[activity.project_id]
                
                # 선행 작업 완료 체크
                all_predecessors_done = True
                earliest_start = project.release_time
                
                for pred_id in activity.predecessors:
                    if pred_id not in scheduled_activities:
                        # 선행 작업이 아직 스케줄되지 않음
                        all_predecessors_done = False
                        break
                    else:
                        # 선행 작업의 완료 시간 고려
                        pred_end_time = schedule[pred_id][1]
                        earliest_start = max(earliest_start, pred_end_time)
                
                # 선행 작업이 완료되지 않았으면 skip
                if not all_predecessors_done:
                    continue
                
                # 동시 수행 불가 제약 확인 (mutually exclusive)
                for mutex_id in activity.mutually_exclusive:
                    if mutex_id in scheduled_activities:
                        # 동시 수행 불가인 activity가 이미 스케줄되어 있으면
                        # 그 activity가 끝난 후에 시작해야 함
                        mutex_end_time = schedule[mutex_id][1]
                        earliest_start = max(earliest_start, mutex_end_time)
                
                # 팀의 가용 시간 확인
                earliest_start = max(earliest_start, team_available_time[team_id])
                
                # 스케줄에 추가
                start_time = earliest_start
                end_time = start_time + activity.duration
                schedule[act_id] = (start_time, end_time, team_id)
                scheduled_activities.add(act_id)
                scheduled_in_this_pass = True
                
                # 팀 가용 시간 업데이트
                team_available_time[team_id] = end_time
                
                # ★ 즉시 처음부터 다시 시작 (for 루프 중단)
                break
            
            # 이번 pass에서 아무것도 스케줄되지 않았으면 종료
            if not scheduled_in_this_pass:
                break
        
        return schedule
    
    def calculate_objective(self, solution: Solution) -> float:
        """
        목적함수 계산: 프로젝트별 tardiness(지연 시간) 합 최소화
        Tardiness = max(0, 완료시간 - 납기)
        스케줄되지 않은 activity가 있으면 큰 penalty
        
        Returns: objective value (작을수록 좋음)
        """
        schedule = self.decode_and_schedule(solution)
        solution.schedule = schedule
        
        # 모든 activity가 스케줄되었는지 확인
        scheduled_activities = set(schedule.keys())
        all_activities = set(act.id for act in self.activities)
        unscheduled_activities = all_activities - scheduled_activities
        
        if len(unscheduled_activities) > 0:
            # 스케줄이 불완전하면 큰 penalty
            objective = len(unscheduled_activities) * 100000
        else:
            # 모든 activity가 스케줄된 경우
            # 각 프로젝트의 완료 시간 계산
            project_completion = {}
            for act_id, (start, end, team_id) in schedule.items():
                proj_id = self.activity_to_project[act_id]
                if proj_id not in project_completion:
                    project_completion[proj_id] = end
                else:
                    project_completion[proj_id] = max(project_completion[proj_id], end)
            
            # 각 프로젝트의 tardiness 계산 및 합산
            total_tardiness = 0
            for proj_id, completion_time in project_completion.items():
                project = self.project_dict[proj_id]
                tardiness = max(0, completion_time - project.due_date)
                total_tardiness += tardiness
            
            # 목적함수: 모든 프로젝트의 tardiness 합 (작을수록 좋음)
            objective = total_tardiness
        
        return objective
    
    def evaluate_population(self, population: List[Solution]):
        """
        Population의 모든 solution에 대해 objective와 fitness 계산
        
        Objective: 작을수록 좋은 값 (팀별 makespan 합)
        Fitness: 클수록 좋은 값 (룰렛 휠 선택용)
                fitness = max_obj - obj + epsilon
        """
        # 1단계: 모든 solution의 objective 계산
        objectives = []
        for solution in population:
            obj = self.calculate_objective(solution)
            solution.objective = obj  # objective 저장
            objectives.append(obj)
        
        # 2단계: Fitness 변환 (큰 값이 좋도록)
        max_obj = max(objectives)
        epsilon = 1.0  # 작은 값 추가 (모든 fitness가 양수가 되도록)
        
        for solution in population:
            # fitness = max_obj - obj + epsilon (클수록 좋음)
            solution.fitness = max_obj - solution.objective + epsilon
    
    def roulette_wheel_selection(self, population: List[Solution]) -> Solution:
        """
        룰렛 휠 선택 (Roulette Wheel Selection)
        Fitness가 클수록 선택될 확률이 높음
        """
        # 전체 fitness 합계 계산
        total_fitness = sum(sol.fitness for sol in population)
        
        # fitness 합이 0 이하인 경우 (모든 solution이 동일하게 나쁜 경우)
        if total_fitness <= 0:
            return random.choice(population)
        
        # 0~total_fitness 사이의 랜덤 값 생성
        pick = random.uniform(0, total_fitness)
        
        # 누적 fitness로 선택
        current = 0
        for solution in population:
            current += solution.fitness
            if current >= pick:
                return solution
        
        # 혹시 모를 경우를 대비해 마지막 solution 반환
        return population[-1]
    
    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """
        Random Key에 대한 crossover (Uniform Crossover)
        각 페어의 random key를 부모 중 하나에서 선택
        """
        child1 = Solution(self.pairs)
        child2 = Solution(self.pairs)
        
        for i in range(len(self.pairs)):
            if random.random() < 0.5:
                child1.random_keys[i] = parent1.random_keys[i]
                child2.random_keys[i] = parent2.random_keys[i]
            else:
                child1.random_keys[i] = parent2.random_keys[i]
                child2.random_keys[i] = parent1.random_keys[i]
        
        return child1, child2
    
    def mutate(self, solution: Solution):
        """
        Random Key에 대한 변이
        랜덤하게 선택된 몇 개의 페어에 대해 새로운 랜덤 키 할당
        """
        num_mutations = max(1, int(len(self.pairs) * 0.1))  # 전체의 10% 변이
        
        for _ in range(num_mutations):
            idx = random.randint(0, len(self.pairs) - 1)
            solution.random_keys[idx] = random.random()  # 새로운 랜덤 키 할당
    
    def evolve(self) -> Solution:
        """메인 진화 루프"""
        print("유전 알고리즘 시작...")
        print(f"Population size: {self.population_size}, Generations: {self.generations}")
        print(f"Activities: {len(self.activities)}, Teams: {self.num_teams}")
        print(f"목적함수: 프로젝트별 지연(tardiness) 시간 합 최소화")
        print(f"선택 방법: Roulette Wheel Selection")
        print(f"디코딩 모드: {self.decode_mode}")
        if self.decode_mode == "batch":
            print("  - Batch: 모든 페어를 순회한 후 다시 처음부터")
        elif self.decode_mode == "immediate":
            print("  - Immediate: activity 스케줄 시 즉시 처음부터 재시작")
        print("-" * 60)
        
        # 초기 population 생성
        population = self.initialize_population()
        self.evaluate_population(population)
        
        # 최적 solution 초기화 (objective가 작을수록 좋음)
        self.best_solution = min(population, key=lambda x: x.objective).copy()
        self.best_fitness_history.append(self.best_solution.objective)
        
        print(f"세대 0: Best Objective = {self.best_solution.objective:.2f}")
        
        # 진화 루프
        for generation in range(1, self.generations + 1):
            new_population = []
            
            # Elitism: 최고의 solution 보존 (objective가 가장 작은 것)
            elite = min(population, key=lambda x: x.objective).copy()
            new_population.append(elite)
            
            # 새로운 population 생성
            while len(new_population) < self.population_size:
                # 선택 (룰렛 휠)
                parent1 = self.roulette_wheel_selection(population)
                parent2 = self.roulette_wheel_selection(population)
                
                # 교차
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 변이
                if random.random() < self.mutation_rate:
                    self.mutate(child1)
                if random.random() < self.mutation_rate:
                    self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Population 크기 조정
            population = new_population[:self.population_size]
            
            # 평가
            self.evaluate_population(population)
            
            # 최적 solution 업데이트 (objective가 작을수록 좋음)
            current_best = min(population, key=lambda x: x.objective)
            if current_best.objective < self.best_solution.objective:
                self.best_solution = current_best.copy()
            
            self.best_fitness_history.append(self.best_solution.objective)
            
            # 진행상황 출력
            if generation % 50 == 0 or generation == self.generations:
                avg_obj = sum(s.objective for s in population) / len(population)
                print(f"세대 {generation}: Best Obj = {self.best_solution.objective:.2f}, "
                      f"Avg Obj = {avg_obj:.2f}")
        
        print("-" * 60)
        print(f"최적화 완료! 최종 Objective = {self.best_solution.objective:.2f}")
        
        return self.best_solution
    
    def print_solution(self, solution: Solution):
        """Solution 정보 출력"""
        print("\n" + "=" * 60)
        print("최적 Solution 상세 정보")
        print("=" * 60)
        
        print(f"\n[목적함수 값]")
        print(f"총 Tardiness (지연 시간 합): {solution.objective:.2f}")
        
        # 프로젝트별 tardiness 계산
        print("\n[프로젝트별 Tardiness]")
        project_completion = {}
        for act_id, (start, end, team_id) in solution.schedule.items():
            proj_id = self.activity_to_project[act_id]
            if proj_id not in project_completion:
                project_completion[proj_id] = end
            else:
                project_completion[proj_id] = max(project_completion[proj_id], end)
        
        for proj_id in sorted(project_completion.keys()):
            project = self.project_dict[proj_id]
            completion = project_completion[proj_id]
            tardiness = max(0, completion - project.due_date)
            status = "지연" if tardiness > 0 else "정시"
            print(f"프로젝트 {proj_id}: 완료={completion}, Due={project.due_date}, "
                  f"Tardiness={tardiness}, 상태={status}")        
        
        # 팀별 할당 정보
        print("\n[팀별 할당된 Activities]")
        team_activities = {i: [] for i in range(self.num_teams)}
        team_makespan = [0] * self.num_teams
        
        for act_id, (start, end, team_id) in solution.schedule.items():
            team_activities[team_id].append(act_id)
            team_makespan[team_id] = max(team_makespan[team_id], end)
        
        for team_id in range(self.num_teams):
            activities = sorted(team_activities[team_id])
            print(f"Team {team_id}: {len(activities)}개 activities, Makespan={team_makespan[team_id]} - {activities}")
        
        print("\n[Schedule - 시간순]")
        # 전체 스케줄을 시간순으로 정렬
        all_schedule = []
        for act_id, (start, end, team_id) in solution.schedule.items():
            all_schedule.append((start, end, act_id, team_id))
        all_schedule.sort(key=lambda x: x[0])
        
        print(f"{'시작':<8} {'종료':<8} {'Activity':<12} {'Team':<8} {'Duration':<10}")
        print("-" * 60)
        for start, end, act_id, team_id in all_schedule:
            activity = self.activity_dict[act_id]
            print(f"{start:<8} {end:<8} Activity {act_id:<4} Team {team_id:<4} "
                  f"(dur={activity.duration})")
        
        print("\n[프로젝트별 상세 스케줄]")
        # 프로젝트별로 정리
        for project in self.projects:
            print(f"\n프로젝트 {project.id} (Release: {project.release_time}, "
                  f"Due: {project.due_date}):")
            
            project_schedule = []
            for activity in project.activities:
                if activity.id in solution.schedule:
                    start, end, team_id = solution.schedule[activity.id]
                    project_schedule.append((activity.id, start, end, team_id))
                else:
                    print(f"  [경고] Activity {activity.id}가 스케줄되지 않음!")
            
            project_schedule.sort(key=lambda x: x[1])  # start time 기준 정렬
            
            for act_id, start, end, team_id in project_schedule:
                activity = self.activity_dict[act_id]
                pred_str = f"선행: {activity.predecessors}" if activity.predecessors else "선행: 없음"
                mutex_str = f", 동시불가: {activity.mutually_exclusive}" if activity.mutually_exclusive else ""
                print(f"  Activity {act_id}: Team {team_id}, "
                      f"Start={start}, End={end}, Duration={activity.duration}, {pred_str}{mutex_str}")
            
            # 프로젝트 완료 시간
            if project_schedule:
                completion = max(x[2] for x in project_schedule)
                tardiness = max(0, completion - project.due_date)
                status = "지연" if tardiness > 0 else "정시"
                print(f"  >> 완료 시간: {completion} (Due: {project.due_date}, 상태: {status}, 지연: {tardiness})")
        
        print("\n" + "=" * 60)


# 예제 실행
if __name__ == "__main__":
    # 예제 데이터 생성
    print("=" * 60)
    print("예제 프로젝트 생성 중...")
    print("=" * 60)
    
    # 프로젝트 1
    project1_activities = [
        Activity(id=1, project_id=1, duration=4, eligible_teams=[0, 1], predecessors=[], 
                 mutually_exclusive=[9]),
        Activity(id=2, project_id=1, duration=3, eligible_teams=[1, 2], predecessors=[1],
                 mutually_exclusive=[3]),
        Activity(id=3, project_id=1, duration=5, eligible_teams=[0, 2, 3], predecessors=[1],
                 mutually_exclusive=[2]),
        Activity(id=4, project_id=1, duration=2, eligible_teams=[2, 3], predecessors=[2, 3],
                 mutually_exclusive=[]),
        Activity(id=5, project_id=1, duration=4, eligible_teams=[0, 1, 3], predecessors=[4],
                 mutually_exclusive=[12]),
    ]
    project1 = Project(id=1, activities=project1_activities, release_time=0, due_date=16)
    
    # 프로젝트 2
    project2_activities = [
        Activity(id=6, project_id=2, duration=3, eligible_teams=[1, 2], predecessors=[],
                 mutually_exclusive=[]),
        Activity(id=7, project_id=2, duration=4, eligible_teams=[0, 2, 3], predecessors=[6],
                 mutually_exclusive=[8]),
        Activity(id=8, project_id=2, duration=3, eligible_teams=[1, 3], predecessors=[6],
                 mutually_exclusive=[7]),
        Activity(id=9, project_id=2, duration=5, eligible_teams=[0, 1], predecessors=[7],
                 mutually_exclusive=[1]),
        Activity(id=10, project_id=2, duration=2, eligible_teams=[0, 2, 3], predecessors=[8, 9],
                 mutually_exclusive=[]),
    ]
    project2 = Project(id=2, activities=project2_activities, release_time=0, due_date=15)
    
    # 프로젝트 3
    project3_activities = [
        Activity(id=11, project_id=3, duration=6, eligible_teams=[0, 2], predecessors=[],
                 mutually_exclusive=[]),
        Activity(id=12, project_id=3, duration=3, eligible_teams=[1, 2, 3], predecessors=[11],
                 mutually_exclusive=[5, 13]),
        Activity(id=13, project_id=3, duration=4, eligible_teams=[0, 1], predecessors=[11],
                 mutually_exclusive=[12]),
        Activity(id=14, project_id=3, duration=3, eligible_teams=[2, 3], predecessors=[12, 13],
                 mutually_exclusive=[19]),
    ]
    project3 = Project(id=3, activities=project3_activities, release_time=2, due_date=18)
    
    # 프로젝트 4
    project4_activities = [
        Activity(id=15, project_id=4, duration=5, eligible_teams=[1, 3], predecessors=[],
                 mutually_exclusive=[]),
        Activity(id=16, project_id=4, duration=4, eligible_teams=[0, 2], predecessors=[15],
                 mutually_exclusive=[17]),
        Activity(id=17, project_id=4, duration=3, eligible_teams=[1, 2, 3], predecessors=[15],
                 mutually_exclusive=[16]),
        Activity(id=18, project_id=4, duration=2, eligible_teams=[0, 1], predecessors=[16],
                 mutually_exclusive=[]),
        Activity(id=19, project_id=4, duration=4, eligible_teams=[2, 3], predecessors=[17],
                 mutually_exclusive=[14]),
        Activity(id=20, project_id=4, duration=3, eligible_teams=[0, 1, 2], predecessors=[18, 19],
                 mutually_exclusive=[]),
    ]
    project4 = Project(id=4, activities=project4_activities, release_time=3, due_date=20)
    
    # 프로젝트 5
    project5_activities = [
        Activity(id=21, project_id=5, duration=4, eligible_teams=[0, 1, 2], predecessors=[],
                 mutually_exclusive=[]),
        Activity(id=22, project_id=5, duration=5, eligible_teams=[1, 3], predecessors=[21],
                 mutually_exclusive=[23]),
        Activity(id=23, project_id=5, duration=3, eligible_teams=[0, 2], predecessors=[21],
                 mutually_exclusive=[22]),
        Activity(id=24, project_id=5, duration=4, eligible_teams=[2, 3], predecessors=[22],
                 mutually_exclusive=[]),
        Activity(id=25, project_id=5, duration=2, eligible_teams=[0, 1, 3], predecessors=[23, 24],
                 mutually_exclusive=[]),
    ]
    project5 = Project(id=5, activities=project5_activities, release_time=1, due_date=17)
    
    projects = [project1, project2, project3, project4, project5]
    num_teams = 4
    
    print(f"\n[데이터 규모]")
    print(f"- 프로젝트 수: {len(projects)}")
    print(f"- 총 Activity 수: {sum(len(p.activities) for p in projects)}")
    print(f"- 팀 수: {num_teams}")
    print(f"\n[동시 수행 불가 제약]")
    mutex_pairs = set()
    for proj in projects:
        for act in proj.activities:
            for mutex_id in act.mutually_exclusive:
                pair = tuple(sorted([act.id, mutex_id]))
                mutex_pairs.add(pair)
    for pair in sorted(mutex_pairs):
        print(f"- Activity {pair[0]} ↔ Activity {pair[1]}")
    print()
    
    # 디코딩 모드 선택
    # "batch": 모든 페어를 순회한 후 다시 처음부터 (기본)
    # "immediate": activity 스케줄 시 즉시 처음부터 재시작
    decode_mode = "immediate"  # 또는 "immediate"
    
    # GA 실행
    ga = GeneticAlgorithm(
        projects=projects,
        num_teams=num_teams,
        population_size=150,  # 문제 크기에 맞춰 증가
        generations=500,  # 세대 수 증가
        crossover_rate=0.8,
        mutation_rate=0.2,
        tournament_size=5,
        decode_mode=decode_mode  # 디코딩 모드 지정
    )
    
    best_solution = ga.evolve()
    ga.print_solution(best_solution)
