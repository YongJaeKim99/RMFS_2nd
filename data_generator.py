"""
RCMPSP (Resource-Constrained Multi-Project Scheduling Problem) 데이터 생성기
프로젝트, Activity, 팀, 제약 조건 등을 생성
"""

import numpy as np
import random
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Activity:
    """Activity 정보"""
    id: int
    project_id: int
    duration: int
    eligible_teams: List[int]
    predecessors: List[int]
    mutually_exclusive: List[int]


@dataclass
class Project:
    """프로젝트 정보"""
    id: int
    activities: List[Activity]
    release_time: int
    due_date: int


def generate_scheduling_data_batch(env_params):
    """
    RCMPSP (Resource-Constrained Multi-Project Scheduling Problem) 데이터 배치 생성
    
    Returns:
        problem: dict containing all problem data as tensors
    """
    batch_size = env_params['batch_size']
    pomo_size = env_params['pomo_size']
    
    # 문제 파라미터
    N_P = env_params['N_P']  # 프로젝트 수
    N_A_min = env_params['N_A_min']  # 프로젝트당 최소 activity 수
    N_A_max = env_params['N_A_max']  # 프로젝트당 최대 activity 수
    N_T = env_params['N_T']  # 팀 수
    duration_min = env_params['duration_min']  # 최소 작업 시간
    duration_max = env_params['duration_max']  # 최대 작업 시간
    
    # 제약 조건 생성 확률
    precedence_prob = env_params.get('precedence_prob', 0.3)  # 선행 관계 생성 확률
    mutex_prob = env_params.get('mutex_prob', 0.1)  # 동시 불가 생성 확률
    eligible_teams_ratio = env_params.get('eligible_teams_ratio', 0.6)  # 평균 eligible 팀 비율
    
    # Due date tightness (1.0 = critical path 길이, 1.5 = 50% 여유)
    due_date_tightness = env_params.get('due_date_tightness', 1.3)
    
    # 배치 데이터 저장
    batch_projects = []
    batch_activities_list = []
    max_activities = 0
    
    for b in range(batch_size):
        projects = []
        all_activities = []
        act_id_counter = 0
        
        for p in range(N_P):
            # 프로젝트당 activity 수 결정
            n_activities = random.randint(N_A_min, N_A_max)
            
            # Activity 생성
            activities = []
            for a in range(n_activities):
                # Duration 생성
                duration = random.randint(duration_min, duration_max)
                
                # Eligible teams 생성 (평균 eligible_teams_ratio 비율)
                num_eligible = max(1, int(N_T * eligible_teams_ratio))
                num_eligible = random.randint(max(1, num_eligible - 1), min(N_T, num_eligible + 1))
                eligible_teams = sorted(random.sample(range(N_T), num_eligible))
                
                # Predecessors (프로젝트 내에서만)
                predecessors = []
                if a > 0 and random.random() < precedence_prob:
                    # 이전 activity 중 일부를 선행 작업으로 지정
                    num_preds = random.randint(1, min(2, a))  # 최대 2개
                    pred_indices = random.sample(range(a), num_preds)
                    predecessors = [act_id_counter - a + idx for idx in pred_indices]
                
                activity = Activity(
                    id=act_id_counter,
                    project_id=p,
                    duration=duration,
                    eligible_teams=eligible_teams,
                    predecessors=predecessors,
                    mutually_exclusive=[]  # 나중에 추가
                )
                activities.append(activity)
                all_activities.append(activity)
                act_id_counter += 1
            
            # 프로젝트 release time (초기 프로젝트는 0, 이후는 랜덤)
            if p == 0:
                release_time = 0
            else:
                release_time = random.randint(0, 5)
            
            # Critical path 계산 (대략적)
            max_path_length = sum(act.duration for act in activities) // max(1, len(activities) // 2)
            due_date = int(release_time + max_path_length * due_date_tightness)
            
            project = Project(
                id=p,
                activities=activities,
                release_time=release_time,
                due_date=due_date
            )
            projects.append(project)
        
        # Mutually exclusive 제약 추가 (프로젝트 간에도 가능)
        for i, act_i in enumerate(all_activities):
            for j in range(i + 1, len(all_activities)):
                # 같은 프로젝트 내에서는 선행 관계가 있으면 mutex 생성 안 함
                if act_i.project_id == all_activities[j].project_id:
                    if all_activities[j].id in act_i.predecessors:
                        continue
                
                # mutex_prob 확률로 동시 불가 제약 생성
                if random.random() < mutex_prob:
                    act_i.mutually_exclusive.append(all_activities[j].id)
                    all_activities[j].mutually_exclusive.append(act_i.id)
        
        batch_projects.append(projects)
        batch_activities_list.append(all_activities)
        max_activities = max(max_activities, len(all_activities))
    
    # Tensor로 변환
    # Activity 속성들을 tensor로 변환
    activity_duration = torch.zeros(batch_size, max_activities, dtype=torch.float)
    activity_project = torch.full((batch_size, max_activities), -1, dtype=torch.long)
    activity_eligible_teams = torch.zeros(batch_size, max_activities, N_T, dtype=torch.bool)
    
    # 제약 관계는 adjacency matrix로 표현
    max_preds = 5  # 최대 선행 작업 수
    activity_predecessors = torch.full((batch_size, max_activities, max_preds), -1, dtype=torch.long)
    max_mutex = 10  # 최대 동시 불가 작업 수
    activity_mutex = torch.full((batch_size, max_activities, max_mutex), -1, dtype=torch.long)
    
    # 프로젝트 정보
    project_release_time = torch.zeros(batch_size, N_P, dtype=torch.float)
    project_due_date = torch.zeros(batch_size, N_P, dtype=torch.float)
    
    # 배치 내 실제 activity 수
    num_activities_per_batch = torch.zeros(batch_size, dtype=torch.long)
    
    for b in range(batch_size):
        projects = batch_projects[b]
        activities = batch_activities_list[b]
        num_activities_per_batch[b] = len(activities)
        
        # Activity 데이터 채우기
        for a_idx, act in enumerate(activities):
            activity_duration[b, a_idx] = act.duration
            activity_project[b, a_idx] = act.project_id
            
            # Eligible teams (one-hot)
            for team in act.eligible_teams:
                activity_eligible_teams[b, a_idx, team] = True
            
            # Predecessors
            for p_idx, pred in enumerate(act.predecessors[:max_preds]):
                activity_predecessors[b, a_idx, p_idx] = pred
            
            # Mutually exclusive
            for m_idx, mutex in enumerate(act.mutually_exclusive[:max_mutex]):
                activity_mutex[b, a_idx, m_idx] = mutex
        
        # 프로젝트 데이터 채우기
        for p_idx, proj in enumerate(projects):
            project_release_time[b, p_idx] = proj.release_time
            project_due_date[b, p_idx] = proj.due_date
    
    # 환경 파라미터
    env_params_tensor = {
        'N_P': N_P,
        'max_N_A': max_activities,
        'N_T': N_T,
        'batch_size': batch_size,
        'pomo_size': pomo_size,
        'objective': env_params.get('objective', 'tardiness'),
    }
    
    problem = {
        'activity_duration': activity_duration,  # (batch_size, max_activities)
        'activity_project': activity_project,  # (batch_size, max_activities)
        'activity_eligible_teams': activity_eligible_teams,  # (batch_size, max_activities, N_T)
        'activity_predecessors': activity_predecessors,  # (batch_size, max_activities, max_preds)
        'activity_mutex': activity_mutex,  # (batch_size, max_activities, max_mutex)
        'project_release_time': project_release_time,  # (batch_size, N_P)
        'project_due_date': project_due_date,  # (batch_size, N_P)
        'num_activities': num_activities_per_batch,  # (batch_size,)
        'env_params': env_params_tensor,
        'batch_projects': batch_projects,  # 원본 데이터 (디버깅용)
        'batch_activities': batch_activities_list,  # 원본 데이터 (디버깅용)
    }
    
    # POMO: 각 텐서를 repeat_interleave로 확장
    for k, v in problem.items():
        if isinstance(v, torch.Tensor) and v.size(0) == batch_size:
            problem[k] = v.repeat_interleave(pomo_size, dim=0)
    
    # batch_projects와 batch_activities도 POMO만큼 복제
    if pomo_size > 1:
        problem['batch_projects'] = [proj for proj in batch_projects for _ in range(pomo_size)]
        problem['batch_activities'] = [acts for acts in batch_activities_list for _ in range(pomo_size)]
    
    # effective batch size 업데이트
    effective_batch_size = batch_size * pomo_size
    problem['env_params']['batch_size'] = effective_batch_size
    
    return problem


def print_problem_summary(problem):
    """문제 요약 출력"""
    env_params = problem['env_params']
    batch_size = env_params['batch_size']
    
    print("\n" + "=" * 60)
    print("Problem Summary")
    print("=" * 60)
    print(f"Batch Size (with POMO): {batch_size}")
    print(f"Number of Projects: {env_params['N_P']}")
    print(f"Max Activities: {env_params['max_N_A']}")
    print(f"Number of Teams: {env_params['N_T']}")
    
    # 첫 번째 배치의 통계
    if 'batch_projects' in problem:
        first_projects = problem['batch_projects'][0]
        first_activities = problem['batch_activities'][0]
        
        print(f"\nFirst Instance Statistics:")
        print(f"  - Total Activities: {len(first_activities)}")
        print(f"  - Projects:")
        for proj in first_projects:
            print(f"    Project {proj.id}: {len(proj.activities)} activities, "
                  f"Release={proj.release_time}, Due={proj.due_date}")
        
        # 제약 통계
        num_preds = sum(len(act.predecessors) for act in first_activities)
        num_mutex = sum(len(act.mutually_exclusive) for act in first_activities) // 2  # 양방향이므로 /2
        print(f"  - Precedence constraints: {num_preds}")
        print(f"  - Mutually exclusive pairs: {num_mutex}")
    
    print("=" * 60 + "\n")


def convert_problem_to_ga_format(problem, batch_idx, num_teams):
    """
    Pickle 데이터를 GA (유전 알고리즘) 형식으로 변환
    
    Args:
        problem: generate_scheduling_data_batch()로 생성된 pickle 데이터
        batch_idx: 추출할 배치 인덱스
        num_teams: 팀 수
    
    Returns:
        List[Project]: GA.py의 Project 객체 리스트
    """
    from GA import Project as GAProject, Activity as GAActivity
    
    # 배치 데이터 추출
    activity_duration = problem['activity_duration'][batch_idx]  # (max_N_A,)
    activity_project = problem['activity_project'][batch_idx]  # (max_N_A,)
    activity_eligible_teams = problem['activity_eligible_teams'][batch_idx]  # (max_N_A, N_T)
    activity_predecessors = problem['activity_predecessors'][batch_idx]  # (max_N_A, max_preds)
    activity_mutex = problem['activity_mutex'][batch_idx]  # (max_N_A, max_mutex)
    project_release_time = problem['project_release_time'][batch_idx]  # (N_P,)
    project_due_date = problem['project_due_date'][batch_idx]  # (N_P,)
    num_activities = problem['num_activities'][batch_idx].item()  # 실제 activity 수
    
    N_P = project_release_time.shape[0]
    
    # 프로젝트별로 Activity 그룹화
    ga_projects = []
    
    for p in range(N_P):
        # 이 프로젝트에 속한 activity 찾기
        proj_mask = (activity_project[:num_activities] == p)
        proj_activity_ids = proj_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        
        if not proj_activity_ids:
            continue
        
        # GA Activity 생성
        ga_activities = []
        for act_id in proj_activity_ids:
            # Eligible teams 추출
            eligible_teams = activity_eligible_teams[act_id].nonzero(as_tuple=False).squeeze(-1).tolist()
            
            # Predecessors 추출 (유효한 것만)
            predecessors = activity_predecessors[act_id]
            valid_preds = predecessors[predecessors >= 0].tolist()
            
            # Mutex activities 추출 (유효한 것만)
            mutex_activities = activity_mutex[act_id]
            valid_mutex = mutex_activities[mutex_activities >= 0].tolist()
            
            ga_activity = GAActivity(
                id=act_id,
                project_id=p,
                duration=int(activity_duration[act_id].item()),
                eligible_teams=eligible_teams,
                predecessors=valid_preds,
                mutually_exclusive=valid_mutex
            )
            ga_activities.append(ga_activity)
        
        # GA Project 생성
        ga_project = GAProject(
            id=p,
            activities=ga_activities,
            release_time=int(project_release_time[p].item()),
            due_date=int(project_due_date[p].item())
        )
        ga_projects.append(ga_project)
    
    return ga_projects


if __name__ == "__main__":
    # 테스트 데이터 생성
    env_params = {
        'batch_size': 2,
        'pomo_size': 4,
        'N_P': 5,  # 프로젝트 수
        'N_A_min': 4,  # 프로젝트당 최소 activity 수
        'N_A_max': 6,  # 프로젝트당 최대 activity 수
        'N_T': 4,  # 팀 수
        'duration_min': 2,
        'duration_max': 6,
        'precedence_prob': 0.3,
        'mutex_prob': 0.1,
        'eligible_teams_ratio': 0.6,
        'due_date_tightness': 1.3,
    }
    
    print("데이터 생성 테스트...")
    problem = generate_scheduling_data_batch(env_params)
    print_problem_summary(problem)
    
    print("✅ 데이터 생성 성공!")
