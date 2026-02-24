"""
간트차트 생성 모듈
Plotly를 사용하여 RCMPSP 스케줄링 결과를 시각화
"""

import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Tuple, List, Set
import numpy as np
import networkx as nx


def create_gantt_chart_from_schedule(
    schedule: Dict[int, Tuple[int, int, int]],
    activity_to_project: Dict[int, int],
    num_teams: int,
    instance_name: str,
    algorithm: str,
    objective_value: float,
    project_due_dates: Dict[int, float] = None,
    save_dir: Path = None,
    show: bool = False
) -> Path:
    """
    스케줄 정보로부터 간트차트 생성
    
    Args:
        schedule: activity_id -> (start_time, end_time, team_id) 매핑
        activity_to_project: activity_id -> project_id 매핑
        num_teams: 팀 수
        instance_name: 인스턴스 이름 (예: "instance_0")
        algorithm: 알고리즘 이름 (예: "GA", "RL")
        objective_value: 목적함수 값
        project_due_dates: project_id -> due_date 매핑 (옵션)
        save_dir: 저장 디렉토리 (None이면 현재 디렉토리/gantt_charts)
        show: 생성 후 브라우저에서 표시할지 여부
    
    Returns:
        저장된 파일 경로
    """
    if save_dir is None:
        save_dir = Path.cwd() / "gantt_charts"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not schedule:
        print(f"⚠️ 스케줄이 비어있어 간트차트를 생성할 수 없습니다: {instance_name}")
        return None
    
    # 프로젝트별 색상 팔레트 정의
    unique_projects = sorted(set(activity_to_project.values()))
    colors = [
        '#FF6B6B',  # 빨강
        '#4ECDC4',  # 청록
        '#45B7D1',  # 하늘
        '#FFA07A',  # 연어
        '#98D8C8',  # 민트
        '#F7DC6F',  # 노랑
        '#BB8FCE',  # 보라
        '#85C1E2',  # 파랑
        '#F8B739',  # 주황
        '#52B788',  # 초록
    ]
    
    project_colors = {}
    for idx, proj_id in enumerate(unique_projects):
        project_colors[proj_id] = colors[idx % len(colors)]
    
    # Figure 생성
    fig = go.Figure()
    
    # 팀별로 데이터 정리
    team_tasks = {i: [] for i in range(num_teams)}
    for act_id, (start, end, team_id) in schedule.items():
        proj_id = activity_to_project[act_id]
        team_tasks[team_id].append({
            'act_id': act_id,
            'start': start,
            'end': end,
            'proj_id': proj_id,
            'duration': end - start
        })
    
    # 각 팀의 activity들을 시작 시간 순으로 정렬
    for team_id in range(num_teams):
        team_tasks[team_id].sort(key=lambda x: x['start'])
    
    # 프로젝트별 완료시간 및 가장 늦게 끝난 activity 계산
    project_completion_info = {}  # proj_id -> (completion_time, latest_act_id, latest_end_time)
    for act_id, (start, end, team_id) in schedule.items():
        proj_id = activity_to_project[act_id]
        if proj_id not in project_completion_info:
            project_completion_info[proj_id] = (end, act_id, end)
        else:
            completion, latest_act, _ = project_completion_info[proj_id]
            if end > completion:
                project_completion_info[proj_id] = (end, act_id, end)
    
    # 팀별로 막대 그래프 추가 (hover 정보 개선)
    for team_id in range(num_teams):
        for task in team_tasks[team_id]:
            proj_id = task['proj_id']
            act_id = task['act_id']
            
            # 해당 프로젝트의 due date 가져오기
            due_date_str = f"{project_due_dates[proj_id]:.1f}" if project_due_dates and proj_id in project_due_dates else "N/A"
            
            # Tardiness 계산 (마지막 activity인 경우만)
            tardiness_str = "N/A"
            if project_due_dates and proj_id in project_completion_info:
                completion, latest_act_id, _ = project_completion_info[proj_id]
                if act_id == latest_act_id and proj_id in project_due_dates:
                    tardiness = max(0, completion - project_due_dates[proj_id])
                    tardiness_str = f"{tardiness:.1f}"
            
            fig.add_trace(go.Bar(
                name=f'Project {proj_id}',
                x=[task['duration']],
                y=[f'Team {team_id}'],
                base=task['start'],
                orientation='h',
                marker=dict(
                    color=project_colors[proj_id],
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    f'<b>Activity {act_id}</b><br>' +
                    f'Project: {proj_id}<br>' +
                    f'Team: {team_id}<br>' +
                    f'Start: {task["start"]:.1f}<br>' +
                    f'End: {task["end"]:.1f}<br>' +
                    f'Duration: {task["duration"]:.1f}<br>' +
                    f'Project Due: {due_date_str}<br>' +
                    f'Tardiness: {tardiness_str}<br>' +
                    '<extra></extra>'
                ),
                legendgroup=f'Project {proj_id}',
                showlegend=False,
                text=f'A{act_id}',
                textposition='inside',
                textfont=dict(size=10, color='white')
            ))
    
    # 범례를 위한 더미 trace 추가 (각 프로젝트별로 한 번만)
    for proj_id in unique_projects:
        fig.add_trace(go.Bar(
            name=f'Project {proj_id}',
            x=[None],
            y=[None],
            marker=dict(color=project_colors[proj_id]),
            legendgroup=f'Project {proj_id}',
            showlegend=True
        ))
    
    # Due date 세로선 추가 (hover 정보 포함)
    if project_due_dates:
        for proj_id in unique_projects:
            due_date = project_due_dates.get(proj_id)
            if due_date is not None:
                # Due date 세로선을 scatter로 추가 (hover 정보를 위해)
                # y축 전체 범위에 걸친 선
                y_range = [f'Team {i}' for i in range(num_teams)]
                
                # 각 y 위치에 점 추가 (선처럼 보이도록)
                fig.add_trace(go.Scatter(
                    x=[due_date] * num_teams,
                    y=y_range,
                    mode='lines',
                    line=dict(
                        color=project_colors[proj_id],
                        width=2,
                        dash='dash'
                    ),
                    opacity=0.7,
                    name=f'P{proj_id} Due',
                    legendgroup=f'due_{proj_id}',
                    showlegend=False,
                    hovertemplate=(
                        f'<b>Project {proj_id} Due Date</b><br>' +
                        f'Due: {due_date:.1f}<br>' +
                        '<extra></extra>'
                    )
                ))
                
                # Due date 라벨 추가 (annotation)
                fig.add_annotation(
                    x=due_date,
                    y=1,
                    yref='paper',
                    text=f"P{proj_id} Due",
                    showarrow=False,
                    yanchor='bottom',
                    font=dict(size=10, color=project_colors[proj_id]),
                    bgcolor='white',
                    bordercolor=project_colors[proj_id],
                    borderwidth=1,
                    borderpad=2
                )
    
    # 레이아웃 설정
    title_text = f'{algorithm} - {instance_name} (Objective: {objective_value:.2f})'
    
    fig.update_layout(
        title=title_text,
        xaxis_title='Time',
        yaxis_title='Team',
        barmode='overlay',
        height=300 + num_teams * 80,  # 팀 수에 따라 높이 조정
        font=dict(size=12),
        showlegend=True,
        hovermode='closest',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            categoryorder='category descending'  # Team 0이 위에 오도록
        ),
        legend=dict(
            title='Projects',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        )
    )
    
    # 파일 저장
    filename = f'{algorithm}_{instance_name}.html'
    filepath = save_dir / filename
    fig.write_html(str(filepath))
    
    print(f"✅ 간트차트 저장: {filepath}")
    
    if show:
        fig.show()
    
    return filepath


def create_gantt_chart_from_env(
    env,
    instance_name: str,
    algorithm: str,
    objective_value: float,
    save_dir: Path = None,
    show: bool = False
) -> Path:
    """
    SchedulingEnv 객체로부터 간트차트 생성
    
    Args:
        env: SchedulingEnv 객체
        instance_name: 인스턴스 이름
        algorithm: 알고리즘 이름
        objective_value: 목적함수 값
        save_dir: 저장 디렉토리
        show: 생성 후 브라우저에서 표시할지 여부
    
    Returns:
        저장된 파일 경로
    """
    # 배치 크기가 1인 경우만 처리
    if env.batch_size != 1:
        print(f"⚠️ 간트차트는 batch_size=1일 때만 생성 가능합니다 (현재: {env.batch_size})")
        return None
    
    # 스케줄 정보 추출
    schedule = {}
    num_act = env.num_activities[0].item()
    
    for act_id in range(num_act):
        if env.activity_started[0, act_id].item():
            start = env.activity_start_time[0, act_id].item()
            end = env.activity_end_time[0, act_id].item()
            team_id = env.activity_assigned_team[0, act_id].item()
            schedule[act_id] = (start, end, team_id)
    
    # activity_to_project 매핑 생성
    activity_to_project = {}
    for act_id in range(num_act):
        proj_id = env.activity_project[0, act_id].item()
        activity_to_project[act_id] = proj_id
    
    # project_due_dates 추출
    project_due_dates = {}
    for proj_id in range(env.N_P):
        due_date = env.project_due_date[0, proj_id].item()
        project_due_dates[proj_id] = due_date
    
    return create_gantt_chart_from_schedule(
        schedule=schedule,
        activity_to_project=activity_to_project,
        num_teams=env.N_T,
        instance_name=instance_name,
        algorithm=algorithm,
        objective_value=objective_value,
        project_due_dates=project_due_dates,
        save_dir=save_dir,
        show=show
    )


def create_gantt_chart_from_ga_solution(
    solution,
    activity_to_project: Dict[int, int],
    num_teams: int,
    instance_name: str,
    objective_value: float = None,
    project_due_dates: Dict[int, float] = None,
    save_dir: Path = None,
    show: bool = False
) -> Path:
    """
    GA Solution 객체로부터 간트차트 생성
    
    Args:
        solution: GA Solution 객체
        activity_to_project: activity_id -> project_id 매핑
        num_teams: 팀 수
        instance_name: 인스턴스 이름
        objective_value: 목적함수 값 (None이면 solution.objective 사용)
        project_due_dates: project_id -> due_date 매핑 (옵션)
        save_dir: 저장 디렉토리
        show: 생성 후 브라우저에서 표시할지 여부
    
    Returns:
        저장된 파일 경로
    """
    if objective_value is None:
        objective_value = solution.objective
    
    return create_gantt_chart_from_schedule(
        schedule=solution.schedule,
        activity_to_project=activity_to_project,
        num_teams=num_teams,
        instance_name=instance_name,
        algorithm="GA",
        objective_value=objective_value,
        project_due_dates=project_due_dates,
        save_dir=save_dir,
        show=show
    )


def create_precedence_graph(
    activity_predecessors: Dict[int, List[int]],
    activity_mutex: Dict[int, List[int]],
    activity_to_project: Dict[int, int],
    instance_name: str,
    activity_eligible_teams: Dict[int, List[int]] = None,
    activity_processing_times: Dict[int, Dict[int, float]] = None,
    save_dir: Path = None,
    show: bool = False
) -> List[Path]:
    """
    Activity 선후관계 그래프 생성 (인스턴스 구조 시각화)
    각 프로젝트별로 독립적인 그래프 파일 생성

    Args:
        activity_predecessors: activity_id -> [predecessor_ids] 매핑
        activity_mutex: activity_id -> [mutex_activity_ids] 매핑
        activity_to_project: activity_id -> project_id 매핑
        instance_name: 인스턴스 이름
        activity_eligible_teams: activity_id -> [eligible_team_ids] 매핑 (옵션)
        activity_processing_times: activity_id -> {team_id: processing_time} 매핑 (옵션)
        save_dir: 저장 디렉토리
        show: 생성 후 브라우저에서 표시할지 여부

    Returns:
        저장된 파일 경로 리스트
    """
    if save_dir is None:
        save_dir = Path.cwd() / "gantt_charts"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 프로젝트별로 그래프 생성
    unique_projects = sorted(set(activity_to_project.values()))
    saved_files = []
    
    for proj_id in unique_projects:
        filepath = _create_single_project_precedence_graph(
            project_id=proj_id,
            activity_predecessors=activity_predecessors,
            activity_mutex=activity_mutex,
            activity_to_project=activity_to_project,
            instance_name=instance_name,
            activity_eligible_teams=activity_eligible_teams,
            activity_processing_times=activity_processing_times,
            save_dir=save_dir,
            show=show
        )
        if filepath:
            saved_files.append(filepath)
    
    return saved_files


def _create_single_project_precedence_graph(
    project_id: int,
    activity_predecessors: Dict[int, List[int]],
    activity_mutex: Dict[int, List[int]],
    activity_to_project: Dict[int, int],
    instance_name: str,
    activity_eligible_teams: Dict[int, List[int]] = None,
    activity_processing_times: Dict[int, Dict[int, float]] = None,
    save_dir: Path = None,
    show: bool = False
) -> Path:
    """
    단일 프로젝트의 선후관계 그래프 생성

    Args:
        project_id: 프로젝트 ID
        activity_predecessors: activity_id -> [predecessor_ids] 매핑
        activity_mutex: activity_id -> [mutex_activity_ids] 매핑
        activity_to_project: activity_id -> project_id 매핑
        instance_name: 인스턴스 이름
        activity_eligible_teams: activity_id -> [eligible_team_ids] 매핑 (옵션)
        activity_processing_times: activity_id -> {team_id: processing_time} 매핑 (옵션)
        save_dir: 저장 디렉토리
        show: 생성 후 브라우저에서 표시할지 여부

    Returns:
        저장된 파일 경로
    """
    # 해당 프로젝트의 activity만 필터링
    project_activities = [act_id for act_id, proj_id in activity_to_project.items() 
                         if proj_id == project_id]
    
    if not project_activities:
        return None
    
    # NetworkX 그래프 생성
    G = nx.DiGraph()
    
    # 해당 프로젝트의 activity를 노드로 추가
    for act_id in project_activities:
        G.add_node(act_id, project=project_id)
    
    # Precedence 관계를 방향 edge로 추가 (같은 프로젝트 내에만)
    precedence_edges = []
    for act_id in project_activities:
        predecessors = activity_predecessors.get(act_id, [])
        for pred_id in predecessors:
            if pred_id >= 0 and pred_id in project_activities:  # 같은 프로젝트 내에만
                G.add_edge(pred_id, act_id, type='precedence')
                precedence_edges.append((pred_id, act_id))
    
    # Mutex 관계를 양방향 edge로 추가 (같은 프로젝트 내에만, 중복 제거)
    mutex_edges = set()
    for act_id in project_activities:
        mutex_list = activity_mutex.get(act_id, [])
        for mutex_id in mutex_list:
            if mutex_id >= 0 and mutex_id in project_activities:  # 같은 프로젝트 내에만
                # 작은 ID를 먼저 오도록 정렬하여 중복 제거
                edge = tuple(sorted([act_id, mutex_id]))
                mutex_edges.add(edge)
    
    # Mutex edge 추가 (양방향)
    for act_id1, act_id2 in mutex_edges:
        G.add_edge(act_id1, act_id2, type='mutex')
        G.add_edge(act_id2, act_id1, type='mutex')
    
    # 프로젝트 색상 (단일 색상)
    colors = [
        '#FF6B6B',  # 빨강
        '#4ECDC4',  # 청록
        '#45B7D1',  # 하늘
        '#FFA07A',  # 연어
        '#98D8C8',  # 민트
        '#F7DC6F',  # 노랑
        '#BB8FCE',  # 보라
        '#85C1E2',  # 파랑
        '#F8B739',  # 주황
        '#52B788',  # 초록
    ]
    project_color = colors[project_id % len(colors)]
    
    # 레이아웃 계산
    if len(project_activities) > 1:
        try:
            # hierarchical layout 시도
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.kamada_kawai_layout(G)
    else:
        # activity가 1개인 경우
        pos = {project_activities[0]: (0, 0)}
    
    # Plotly Figure 생성
    fig = go.Figure()
    
    # Precedence edges 그리기 (검은색 화살표)
    for edge in G.edges():
        if G.edges[edge].get('type') == 'precedence':
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(color='black', width=2),
                hoverinfo='none',
                showlegend=False
            ))
            
            # 화살표 추가
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black'
            )
    
    # Mutex edges 그리기 (빨간색 양방향, 중복 제거)
    drawn_mutex = set()
    for act_id1, act_id2 in mutex_edges:
        if (act_id1, act_id2) not in drawn_mutex and (act_id2, act_id1) not in drawn_mutex:
            x0, y0 = pos[act_id1]
            x1, y1 = pos[act_id2]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # 양방향 화살표 (양쪽 끝에 화살표)
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )
            fig.add_annotation(
                x=x0, y=y0,
                ax=x1, ay=y1,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )
            
            drawn_mutex.add((act_id1, act_id2))
    
    # 노드 그리기 (단일 프로젝트)
    node_x = []
    node_y = []
    node_text = []
    node_hover = []

    for act_id in project_activities:
        x, y = pos[act_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'A{act_id}')

        # Hover 정보 구성
        hover_lines = [f'<b>Activity {act_id}</b>', f'Project: {project_id}']

        if activity_eligible_teams and act_id in activity_eligible_teams:
            teams = activity_eligible_teams[act_id]
            hover_lines.append(f'Eligible Teams: {teams}')

            # 팀별 처리시간 표시
            if activity_processing_times and act_id in activity_processing_times:
                pt = activity_processing_times[act_id]
                for t in teams:
                    if t in pt:
                        hover_lines.append(f'  T{t}: {pt[t]:.0f}')
        elif activity_processing_times and act_id in activity_processing_times:
            pt = activity_processing_times[act_id]
            hover_lines.append('Processing Times:')
            for t, dur in sorted(pt.items()):
                hover_lines.append(f'  T{t}: {dur:.0f}')

        node_hover.append('<br>'.join(hover_lines) + '<extra></extra>')

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=40,
            color=project_color,
            line=dict(color='white', width=2)
        ),
        text=node_text,
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial Black'),
        name=f'Project {project_id}',
        hovertemplate=node_hover
    ))

    # Eligible team & processing time 테이블을 그래프 하단에 annotation으로 추가
    if activity_eligible_teams or activity_processing_times:
        table_lines = []
        for act_id in sorted(project_activities):
            parts = [f'A{act_id}:']
            if activity_eligible_teams and act_id in activity_eligible_teams:
                teams = activity_eligible_teams[act_id]
                if activity_processing_times and act_id in activity_processing_times:
                    pt = activity_processing_times[act_id]
                    team_strs = [f'T{t}({pt[t]:.0f})' if t in pt else f'T{t}' for t in teams]
                else:
                    team_strs = [f'T{t}' for t in teams]
                parts.append(', '.join(team_strs))
            elif activity_processing_times and act_id in activity_processing_times:
                pt = activity_processing_times[act_id]
                team_strs = [f'T{t}({dur:.0f})' for t, dur in sorted(pt.items())]
                parts.append(', '.join(team_strs))
            table_lines.append(' '.join(parts))

        table_text = '<br>'.join(table_lines)
        fig.add_annotation(
            text=f'<b>Eligible Teams (Processing Time)</b><br>{table_text}',
            xref='paper', yref='paper',
            x=0, y=-0.05,
            xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=11, family='Courier New'),
            align='left',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            borderpad=6
        )
    
    # 범례 항목 추가 (edge 타입 설명)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='black', width=2),
        name='Precedence',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Mutex (양방향)',
        showlegend=True
    ))
    
    # 레이아웃 설정
    has_table = activity_eligible_teams or activity_processing_times
    bottom_margin = 30 + len(project_activities) * 18 if has_table else 30
    fig.update_layout(
        title=f'{instance_name} - Project {project_id} - Activity Precedence & Mutex',
        showlegend=True,
        hovermode='closest',
        height=600 + (bottom_margin if has_table else 0),
        margin=dict(b=bottom_margin),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    # 파일 저장
    filename = f'{instance_name}_project{project_id}_precedence.html'
    filepath = save_dir / filename
    fig.write_html(str(filepath))
    
    print(f"✅ 선후관계 그래프 저장: {filepath}")
    
    if show:
        fig.show()
    
    return filepath
