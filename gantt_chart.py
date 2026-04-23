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
    activity_step_order: Dict[int, int] = None,
    objective_type: str = 'tardiness',
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
        activity_step_order: activity_id -> step_number 매핑 (옵션, RL용)
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
            
            # 목적함수별 hover 정보 계산
            obj_hover_str = ""
            if objective_type == 'tardiness':
                tardiness_str = "N/A"
                if project_due_dates and proj_id in project_completion_info:
                    completion, latest_act_id, _ = project_completion_info[proj_id]
                    if act_id == latest_act_id and proj_id in project_due_dates:
                        tardiness = max(0, completion - project_due_dates[proj_id])
                        tardiness_str = f"{tardiness:.1f}"
                obj_hover_str = f'Project Due: {due_date_str}<br>Tardiness: {tardiness_str}<br>'
            elif objective_type == 'makespan':
                if proj_id in project_completion_info:
                    completion, latest_act_id, _ = project_completion_info[proj_id]
                    if act_id == latest_act_id:
                        obj_hover_str = f'Project Completion: {completion:.1f}<br>'

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
                    f'<b>Activity {act_id}</b>' +
                    (f' (Step {activity_step_order[act_id]})' if activity_step_order and act_id in activity_step_order else '') +
                    f'<br>' +
                    f'Project: {proj_id}<br>' +
                    f'Team: {team_id}<br>' +
                    f'Start: {task["start"]:.1f}<br>' +
                    f'End: {task["end"]:.1f}<br>' +
                    f'Duration: {task["duration"]:.1f}<br>' +
                    obj_hover_str +
                    '<extra></extra>'
                ),
                legendgroup=f'Project {proj_id}',
                showlegend=False,
                text=(f'A{act_id}({activity_step_order[act_id]})' if activity_step_order and act_id in activity_step_order else f'A{act_id}'),
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
    title_text = f'{algorithm} - {instance_name} ({objective_type.capitalize()}: {objective_value:.2f})'
    
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
            categoryorder='array',
            categoryarray=[f'Team {i}' for i in range(num_teams - 1, -1, -1)]  # Team 0이 위에 오도록 (숫자 순)
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
    objective_type: str = 'tardiness',
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
        if env.activity_reserved[0, act_id].item():
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
        objective_type=objective_type,
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
    objective_type: str = 'tardiness',
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
        objective_type: 목적함수 종류 ('tardiness' or 'makespan')
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
        objective_type=objective_type,
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
    project_release_times: Dict[int, float] = None,
    project_due_dates: Dict[int, float] = None,
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
        project_release_times: project_id -> release_time 매핑 (옵션)
        project_due_dates: project_id -> due_date 매핑 (옵션)
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
            project_release_time=project_release_times.get(proj_id) if project_release_times else None,
            project_due_date=project_due_dates.get(proj_id) if project_due_dates else None,
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
    project_release_time: float = None,
    project_due_date: float = None,
    save_dir: Path = None,
    show: bool = False
) -> Path:
    """
    단일 프로젝트의 선후관계 DAG 그래프 생성
    왼쪽→오른쪽 레이어드 레이아웃, S/E 더미 노드 포함
    """
    project_activities = sorted([act_id for act_id, proj_id in activity_to_project.items()
                                  if proj_id == project_id])
    project_act_set = set(project_activities)

    if not project_activities:
        return None

    # ── 1) Precedence DAG 구축 (mutex는 별도 처리) ──
    G = nx.DiGraph()
    for act_id in project_activities:
        G.add_node(act_id)

    precedence_edges = []
    for act_id in project_activities:
        for pred_id in activity_predecessors.get(act_id, []):
            if pred_id >= 0 and pred_id in project_act_set:
                G.add_edge(pred_id, act_id)
                precedence_edges.append((pred_id, act_id))

    mutex_edges: Set[Tuple[int, int]] = set()
    for act_id in project_activities:
        for mutex_id in activity_mutex.get(act_id, []):
            if mutex_id >= 0 and mutex_id in project_act_set:
                mutex_edges.add(tuple(sorted([act_id, mutex_id])))

    # ── 2) S / E 더미 노드 추가 ──
    sources = [n for n in project_activities if G.in_degree(n) == 0]
    sinks   = [n for n in project_activities if G.out_degree(n) == 0]

    G.add_node('S')
    G.add_node('E')
    for s in sources:
        G.add_edge('S', s)
    for s in sinks:
        G.add_edge(s, 'E')

    # ── 3) 위상 정렬 기반 레이어 계산 (longest path from S) ──
    topo_order = list(nx.topological_sort(G))
    dist = {n: 0 for n in G.nodes()}
    for n in topo_order:
        for succ in G.successors(n):
            dist[succ] = max(dist[succ], dist[n] + 1)

    max_layer = max(dist.values()) if dist else 1
    layers: Dict[int, list] = {}
    for n, d in dist.items():
        layers.setdefault(d, []).append(n)

    # ── 4) 레이어 내 y-순서 최적화 (median heuristic, 2-pass) ──
    pos: Dict = {}

    # 초기 배치 (ID 순)
    for layer_idx in sorted(layers.keys()):
        nodes = layers[layer_idx]
        nodes.sort(key=lambda n: (-1 if n == 'S' else (10**9 if n == 'E' else n)))
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            y = ((n_nodes - 1) / 2 - i) * 1.5 if n_nodes > 1 else 0
            pos[node] = (layer_idx, y)

    # Forward pass: median of predecessors
    for layer_idx in sorted(layers.keys()):
        if layer_idx == 0:
            continue
        nodes = layers[layer_idx]

        def _median_pred_y(node):
            preds = [p for p in G.predecessors(node) if p in pos]
            if not preds:
                return 0
            ys = sorted(pos[p][1] for p in preds)
            return ys[len(ys) // 2]

        nodes.sort(key=_median_pred_y, reverse=True)
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            y = ((n_nodes - 1) / 2 - i) * 1.5 if n_nodes > 1 else 0
            pos[node] = (layer_idx, y)

    # Backward pass: median of successors
    for layer_idx in sorted(layers.keys(), reverse=True):
        if layer_idx == max_layer:
            continue
        nodes = layers[layer_idx]

        def _median_succ_y(node):
            succs = [s for s in G.successors(node) if s in pos]
            if not succs:
                return 0
            ys = sorted(pos[s][1] for s in succs)
            return ys[len(ys) // 2]

        nodes.sort(key=_median_succ_y, reverse=True)
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            y = ((n_nodes - 1) / 2 - i) * 1.5 if n_nodes > 1 else 0
            pos[node] = (layer_idx, y)

    # ── 5) Plotly Figure 생성 ──
    fig = go.Figure()

    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
    ]
    project_color = colors[project_id % len(colors)]

    # ─ 5a) Precedence edges (solid black) ─
    edge_x, edge_y = [], []
    for pred, succ in precedence_edges:
        x0, y0 = pos[pred]
        x1, y1 = pos[succ]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none', showlegend=False
        ))
    for pred, succ in precedence_edges:
        x0, y0 = pos[pred]
        x1, y1 = pos[succ]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=2, arrowcolor='black'
        )

    # ─ 5b) Dummy edges: S→sources, sinks→E (dotted gray) ─
    dummy_x, dummy_y = [], []
    dummy_arrows: List[Tuple] = []
    for src in sources:
        x0, y0 = pos['S']
        x1, y1 = pos[src]
        dummy_x.extend([x0, x1, None])
        dummy_y.extend([y0, y1, None])
        dummy_arrows.append(('S', src))
    for sink in sinks:
        x0, y0 = pos[sink]
        x1, y1 = pos['E']
        dummy_x.extend([x0, x1, None])
        dummy_y.extend([y0, y1, None])
        dummy_arrows.append((sink, 'E'))
    if dummy_x:
        fig.add_trace(go.Scatter(
            x=dummy_x, y=dummy_y, mode='lines',
            line=dict(color='gray', width=1.5, dash='dot'),
            hoverinfo='none', showlegend=False
        ))
    for src_node, dst_node in dummy_arrows:
        x0, y0 = pos[src_node]
        x1, y1 = pos[dst_node]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor='gray'
        )

    # ─ 5c) Mutex edges (dashed red, bidirectional) ─
    mutex_x, mutex_y = [], []
    for act1, act2 in mutex_edges:
        x0, y0 = pos[act1]
        x1, y1 = pos[act2]
        mutex_x.extend([x0, x1, None])
        mutex_y.extend([y0, y1, None])
    if mutex_x:
        fig.add_trace(go.Scatter(
            x=mutex_x, y=mutex_y, mode='lines',
            line=dict(color='red', width=1.5, dash='dash'),
            hoverinfo='none', showlegend=False
        ))
    for act1, act2 in mutex_edges:
        x0, y0 = pos[act1]
        x1, y1 = pos[act2]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor='red'
        )
        fig.add_annotation(
            x=x0, y=y0, ax=x1, ay=y1,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor='red'
        )

    # ─ 5d) Activity 노드 (프로젝트 색상) ─
    node_x, node_y, node_text, node_hover = [], [], [], []
    for act_id in project_activities:
        x, y = pos[act_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'A{act_id}')

        hover_lines = [f'<b>Activity {act_id}</b>', f'Project: {project_id}']
        if activity_eligible_teams and act_id in activity_eligible_teams:
            teams = activity_eligible_teams[act_id]
            hover_lines.append(f'Eligible Teams: {teams}')
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
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=40, color=project_color, line=dict(color='white', width=2)),
        text=node_text,
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial Black'),
        name=f'Project {project_id}',
        hovertemplate=node_hover
    ))

    # ─ 5e) S / E 더미 노드 (흰색 원, 검은 테두리) ─
    fig.add_trace(go.Scatter(
        x=[pos['S'][0], pos['E'][0]],
        y=[pos['S'][1], pos['E'][1]],
        mode='markers+text',
        marker=dict(size=40, color='white', symbol='circle',
                    line=dict(color='black', width=2)),
        text=['S', 'E'],
        textposition='middle center',
        textfont=dict(size=14, color='black', family='Arial Black'),
        name='Dummy (S/E)',
        hoverinfo='skip'
    ))

    # ─ Eligible team & processing time 테이블 ─
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
            bordercolor='gray', borderwidth=1, borderpad=6
        )

    # ─ 범례 ─
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='black', width=2),
        name='Precedence', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Mutex', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='gray', width=1.5, dash='dot'),
        name='Dummy (S/E)', showlegend=True
    ))

    # ─ 레이아웃 설정 ─
    has_table = activity_eligible_teams or activity_processing_times
    bottom_margin = 30 + len(project_activities) * 18 if has_table else 30

    title_parts = [f'{instance_name} - Project {project_id} - Activity DAG']
    time_info = []
    if project_release_time is not None:
        time_info.append(f'Release: {project_release_time:.0f}')
    if project_due_date is not None:
        time_info.append(f'Due: {project_due_date:.0f}')
    if time_info:
        title_parts.append(f'({", ".join(time_info)})')

    fig.update_layout(
        title=' '.join(title_parts),
        showlegend=True,
        hovermode='closest',
        height=500 + (bottom_margin if has_table else 0),
        margin=dict(b=bottom_margin, l=40, r=40, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    # 파일 저장
    filename = f'{instance_name}_project{project_id}_precedence.html'
    filepath = save_dir / filename
    fig.write_html(str(filepath))

    print(f"  DAG 저장: {filepath}")

    if show:
        fig.show()

    return filepath
