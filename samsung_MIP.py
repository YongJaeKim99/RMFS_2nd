try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict
from ortools.sat.python import cp_model

from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

# ==========================================
# 1. 데이터 클래스 및 인스턴스 생성
# ==========================================

@dataclass
class Instance:
    # 프로젝트 구조
    num_projects: int
    num_companies: int
    set_sizes: List[int] = field(default_factory=lambda: [9, 9, 1, 2, 9, 1])
    
    durations: List[int] = field(default_factory=list)
    precedences: Dict[int, List[int]] = field(default_factory=dict)
    nooverlaps: Dict[int, List[int]] = field(default_factory=dict)
    activity_to_project: Dict[int, int] = field(default_factory=dict)
    project_to_activities: Dict[int, List[int]] = field(default_factory=dict)
    due_dates: Dict[int, int] = field(default_factory=dict) 
    release_times: Dict[int, int] = field(default_factory=dict)
    activity_to_set: Dict[int, int] = field(default_factory=dict)
    set_to_activities: Dict[int, List[int]] = field(default_factory=dict)

    # 자원 구조
    num_teams: int = 0
    activity_to_teams: Dict[int, List[int]] = field(default_factory=dict)
    team_to_company: Dict[int, int] = field(default_factory=dict)
    company_to_teams: Dict[int, List[int]] = field(default_factory=dict)
    set_to_companies: Dict[int, List[int]] = field(default_factory=dict)
    project_set_to_companies: Dict[int, Dict[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        self.num_standard_activities = sum(self.set_sizes)
        self.num_activities = self.num_standard_activities * self.num_projects
        self.durations = [0] * self.num_activities

    
def create_random_instance(num_projects=2, num_companies=4, seed=42) -> Instance:
    random.seed(seed)
    inst = Instance(num_projects=num_projects, num_companies=num_companies)

    # Activity 생성
    for i in range(inst.num_activities):
        inst.activity_to_project[i] = i // inst.num_standard_activities

    for p in range(inst.num_projects):
        inst.project_to_activities[p] = [p * inst.num_standard_activities + i for i in range(inst.num_standard_activities)]

    curr_id = 0
    for p in range(inst.num_projects):
        for set_idx, size in enumerate(inst.set_sizes):
            if p == 0:
                inst.set_to_activities[set_idx] = []
            for _ in range(size):
                inst.durations[curr_id] = random.randint(3, 7)
                inst.activity_to_set[curr_id] = set_idx
                inst.set_to_activities[set_idx].append(curr_id)
                curr_id += 1
            
    # 선후행 관계 (제공해주신 자료 그대로 참조)
    standard_precedence = {
        9:[3],
        10:[6,7,8],
        11:[6,7,8],
        12:[6,7,8],
        13:[6,7,8],
        14:[18],
        18:[9,10,11,12,13,15,16,17], # 이 부분은 추정
        19:[0],
        20:[0],
        21:[1],
        22:[2],
        23:[3],
        24:[4],
        25:[5],
        26:[23],
        27:[26],
        30:[14,18,19,20,21,22,24,25,27,28,29] # 이 부분은 추정
    }

    # 중복 금지 관계
    standard_nooverlap = {
        18:[i for i in range(inst.num_standard_activities) if i != 18]
    }
     
    inst.precedences = {i: [] for i in range(inst.num_activities)}
    inst.nooverlaps = {i: [] for i in range(inst.num_activities)}
    for p in range(inst.num_projects):
        addnum = p * inst.num_standard_activities

        for k, values in standard_precedence.items():
            for v in values:
                inst.precedences[k + addnum].append(v + addnum)

        for k, values in standard_nooverlap.items():
            for v in values:
                inst.nooverlaps[k + addnum].append(v + addnum)

    # 업체/팀 생성
    curr_id = 0
    for c_id in range(num_companies):
        num_teams = 3
        if c_id == 1: num_teams = 5 # 첫번째 집합 (도면)은 리소스가 많아 병목이 되지 않는다고 가정
        if c_id == num_companies-1: num_teams = 2 * num_projects # 마지막 회사는 리소스가 무한인 회사

        teams = [curr_id + k for k in range(num_teams)]
        inst.company_to_teams[c_id] = teams
        for t_id in teams:
            inst.team_to_company[t_id] = c_id

        curr_id += num_teams
    inst.num_teams = curr_id

    # 업체 set 할당 NOTE: 가정 - 리소스 안쓰는건 리소스 무한대 회사 하나가 하고, 동시에 두 집합을 하는 회사도 존재
    n, m = num_companies-1, len(inst.set_sizes)-3
    separators = sorted(random.sample(range(1, n), m - 1))
    points = [0] + separators + [n]
    result = [points[i+1] - points[i] for i in range(m)]

    inst.set_to_companies = {}
    curr_id = 0
    prev = 0
    for set_idx in range(len(inst.set_sizes)):
        if set_idx in [2, 5]: # 마지막 회사가 2, 5번 집합 (설비 반입, 셋업)을 한다고 가정
            inst.set_to_companies[set_idx] = [num_companies-1]
            continue
        if set_idx in [4]: # 1번 집합과 4번 집합은 동일한 회사가 한다고 가정
            inst.set_to_companies[set_idx] = inst.set_to_companies[1]
            continue
        inst.set_to_companies[set_idx] = [c + prev for c in range(result[curr_id])]
        prev += result[curr_id]
        curr_id += 1
        
    # 프로젝트 할당
    total_dur_sum = sum(inst.durations)
    for p_id in range(num_projects):
        passigns = {}
        for set_idx in range(len(inst.set_sizes)):
            t_id = random.choice(inst.set_to_companies[set_idx])
            passigns[set_idx] = t_id

        inst.due_dates[p_id] = random.randint(int(total_dur_sum * 0.10), int(total_dur_sum * 0.20))
        inst.release_times[p_id] = 0
        inst.project_set_to_companies[p_id] = passigns

    curr_id = 0
    for p in range(inst.num_projects):
        for set_idx, size in enumerate(inst.set_sizes):
            t_id = inst.project_set_to_companies[p][set_idx]
            for _ in range(size):
                teams = inst.company_to_teams[t_id] # Random eligibility
                inst.activity_to_teams[curr_id] = random.sample(teams, random.randint(1, len(teams)))
                curr_id += 1
        
    return inst


def solve_rcmpsp_gurobi(inst, time_limit: int):
    if not HAS_GUROBI:
        print("Gurobi is not installed. Skipping MIP solve.")
        return None
    print(f"\nSolving RCMPSP (MIP Formulation using Gurobi) for {inst.num_projects} projects...")

    m = gp.Model("MRCPSP_MIP")

    # team-dependent durations: d[i][k]
    # 후방 호환: activity_team_durations가 없으면 durations 사용
    has_team_dur = hasattr(inst, 'activity_team_durations')
    d = {}
    for i in range(inst.num_activities):
        d[i] = {}
        for k in inst.activity_to_teams[i]:
            if has_team_dur:
                d[i][k] = inst.activity_team_durations[i][k]
            else:
                d[i][k] = inst.durations[i]

    M_val = sum(max(d[i].values()) for i in range(inst.num_activities)) * 2

    # --- Variables ---
    # C_i: Completion Time
    C = {}
    for i in range(inst.num_activities):
        C[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_{i}")

    # x_{i,k}: Team Assignment (Binary)
    x = {}
    for i in range(inst.num_activities):
        valid_teams = inst.activity_to_teams[i]
        for t in valid_teams:
            x[i, t] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    # y_{i,j}: Resource Conflict Ordering (Binary)
    y = {}
    resource_conflicts = [] 
    for i in range(inst.num_activities):
        teams_i = set(inst.activity_to_teams[i])
        for j in range(i + 1, inst.num_activities):
            teams_j = set(inst.activity_to_teams[j])
            
            # Intersection check
            common_teams = teams_i.intersection(teams_j)
            if common_teams:
                y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
                for k in common_teams:
                    resource_conflicts.append((i, j, k))

    # z_{i,j}: No-Overlap Ordering (Binary)
    z = {}
    ct_pairs = []
    for i, neighbors in inst.nooverlaps.items():
        for j in neighbors:
            if inst.activity_to_project[i] == inst.activity_to_project[j]:
                u, v = sorted((i, j))
                if u == v:
                    continue

                if (u, v) not in z:
                    z[u, v] = m.addVar(vtype=GRB.BINARY, name=f"z_{u}_{v}")
                    ct_pairs.append((u, v))

    # T_p: Project Tardiness
    Tardy = {}
    for p_id in range(inst.num_projects):
        Tardy[p_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Tardy_{p_id}")

    m.update()

    # --- Constraints ---

    # Eq 2: Assignment (Sum x_{i,k} = 1)
    for i in range(inst.num_activities):
        valid_teams = inst.activity_to_teams[i]
        m.addConstr(gp.quicksum(x[i, k] for k in valid_teams) == 1, name=f"Assign_{i}")

    # Eq 3: Precedence (C_j >= C_i + sum_k(d[j,k] * x[j,k]))
    for j in range(inst.num_activities):
        d_j_expr = gp.quicksum(d[j][k] * x[j, k] for k in inst.activity_to_teams[j])
        for i in inst.precedences.get(j, []):
            m.addConstr(C[j] >= C[i] + d_j_expr, name=f"Pred_{i}_{j}")

    # Eq 4 & 5: Resource Conflict (Big-M) with team-dependent duration
    for i, j, k in resource_conflicts:
        d_jk = d[j][k]
        d_ik = d[i][k]

        # Eq 4: if both on team k and i->j, C_j >= C_i + d[j,k]
        m.addConstr(
            C[j] >= C[i] + d_jk - M_val * (3 - x[i, k] - x[j, k] - y[i, j]),
            name=f"Res_Fwd_{i}_{j}_{k}"
        )

        # Eq 5: if both on team k and j->i, C_i >= C_j + d[i,k]
        m.addConstr(
            C[i] >= C[j] + d_ik - M_val * (2 - x[i, k] - x[j, k] + y[i, j]),
            name=f"Res_Bwd_{i}_{j}_{k}"
        )

    # Eq 6 & 7: No-Overlap (Big-M) with team-dependent duration
    for i, j in ct_pairs:
        d_j_expr = gp.quicksum(d[j][k] * x[j, k] for k in inst.activity_to_teams[j])
        d_i_expr = gp.quicksum(d[i][k] * x[i, k] for k in inst.activity_to_teams[i])

        # Eq 6: if i before j, C_j >= C_i + d_j
        m.addConstr(
            C[j] >= C[i] + d_j_expr - M_val * (1 - z[i, j]),
            name=f"CT_1_{i}_{j}"
        )

        # Eq 7: if j before i, C_i >= C_j + d_i
        m.addConstr(
            C[i] >= C[j] + d_i_expr - M_val * z[i, j],
            name=f"CT_2_{i}_{j}"
        )

    # Eq 8: Release Time (C_i >= sum_k(d[i,k]*x[i,k]) + RT_p)
    for i in range(inst.num_activities):
        p_id = inst.activity_to_project[i]
        rt_p = inst.release_times[p_id]
        d_i_expr = gp.quicksum(d[i][k] * x[i, k] for k in inst.activity_to_teams[i])
        m.addConstr(C[i] >= d_i_expr + rt_p, name=f"Release_{i}")

    # Eq 9: Tardiness (T_p >= C_i - DD_p)
    for i in range(inst.num_activities):
        p_id = inst.activity_to_project[i]
        dd_p = inst.due_dates[p_id]
        m.addConstr(Tardy[p_id] >= C[i] - dd_p, name=f"TardyConst_{i}")

    # Obj: Minimize Sum Tardiness
    m.setObjective(gp.quicksum(Tardy[p] for p in range(inst.num_projects)), GRB.MINIMIZE)
    
    m.setParam('OutputFlag', 1)
    m.setParam(GRB.Param.TimeLimit, time_limit)
    m.optimize()

    # --- Result Retrieval ---
    if m.SolCount > 0:
        if m.Status == GRB.OPTIMAL:
            sol_status = "optimal"
            print(f"Optimal solution found! Objective: {m.ObjVal} Time: {m.Runtime:.2f}s")
        else:
            sol_status = "feasible"
            print(f"Feasible solution found! Objective: {m.ObjVal} Time: {m.Runtime:.2f}s")

        # Recover Assigned Teams
        assigned_teams = {}
        for (i, t), var in x.items():
            if var.X > 0.5:
                assigned_teams[i] = t

        # Recover Start Times (using assigned team's duration)
        start_times = {}
        for i in range(inst.num_activities):
            t = assigned_teams.get(i)
            dur_i = d[i][t] if t is not None else inst.durations[i]
            start_times[i] = C[i].X - dur_i

        return m.ObjVal, start_times, assigned_teams, sol_status

    elif m.Status == GRB.TIME_LIMIT:
        print("Time limit reached with no feasible solution.")
        return None

    else:
        print("Model is infeasible or unbounded.")
        return None


def solve_rcmpsp_cp(inst, time_limit: int):
    """
    Solves the Multi-Mode Resource Constrained Project Scheduling Problem (MRCPSP)
    using Google OR-Tools CP-SAT solver.
    Matches the formulation of the provided Gurobi MIP function.
    """
    print(f"\nSolving RCMPSP (CP-SAT Formulation) for {inst.num_projects} projects...")

    model = cp_model.CpModel()

    # team-dependent durations: d[i][k]
    has_team_dur = hasattr(inst, 'activity_team_durations')
    d = {}
    for i in range(inst.num_activities):
        d[i] = {}
        for k in inst.activity_to_teams[i]:
            if has_team_dur:
                d[i][k] = inst.activity_team_durations[i][k]
            else:
                d[i][k] = inst.durations[i]

    # Horizon Calculation
    horizon = sum(max(d[i].values()) for i in range(inst.num_activities)) * 2

    # --- Variables ---

    intervals = {}        # Master intervals for activities (I_i)
    starts = {}           # Start time variables
    ends = {}             # End time variables
    durations_var = {}    # Duration variables (team-dependent)
    presences = {}        # (i, k) -> BoolVar

    all_teams = set()
    for teams in inst.activity_to_teams.values():
        all_teams.update(teams)
    intervals_per_team = {k: [] for k in all_teams}

    for i in range(inst.num_activities):
        valid_teams = inst.activity_to_teams[i]
        team_durs = [d[i][k] for k in valid_teams]

        start_var = model.NewIntVar(0, horizon, f'start_{i}')
        end_var = model.NewIntVar(0, horizon, f'end_{i}')

        # Duration variable with domain restricted to possible team durations
        if len(set(team_durs)) == 1:
            # All teams have same duration — fixed
            dur_var = team_durs[0]
        else:
            dur_var = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(sorted(set(team_durs))), f'dur_{i}'
            )

        interval_var = model.NewIntervalVar(start_var, dur_var, end_var, f'interval_{i}')

        intervals[i] = interval_var
        starts[i] = start_var
        ends[i] = end_var
        durations_var[i] = dur_var

        team_presences = []

        for t in valid_teams:
            is_present = model.NewBoolVar(f'pres_{i}_{t}')
            presences[(i, t)] = is_present
            team_presences.append(is_present)

            # Link duration to team selection
            if not isinstance(dur_var, int):
                model.Add(dur_var == d[i][t]).OnlyEnforceIf(is_present)

            # Optional interval with team-specific fixed duration for NoOverlap
            opt_interval = model.NewOptionalIntervalVar(
                start_var, d[i][t], end_var, is_present, f'opt_interval_{i}_{t}'
            )
            intervals_per_team[t].append(opt_interval)

        # Exactly one team must be chosen
        model.Add(sum(team_presences) == 1)

    # 2. Project Tardiness Variables (T_p)
    tardy_vars = {}
    for p_id in range(inst.num_projects):
        tardy_vars[p_id] = model.NewIntVar(0, horizon, f'Tardy_{p_id}')

    # --- Constraints ---

    # Eq 3: Precedence Constraints (EndBeforeStart)
    for j in range(inst.num_activities):
        for i in inst.precedences.get(j, []):
            # End(I_i) <= Start(I_j)
            model.Add(starts[j] >= ends[i])

    # Eq 4 & 5: Resource Constraints (NoOverlap)
    # NoOverlap({O_{i,k} | i in A, k in K}) -> For each team, intervals cannot overlap.
    for t, team_intervals in intervals_per_team.items():
        if len(team_intervals) > 0:
            model.AddNoOverlap(team_intervals)

    # Eq 6 & 7: Specific No-Overlap Constraints (Disjunctive Pairs)
    # The Gurobi code loops through inst.nooverlaps and adds constraints for pairs in the same project.
    added_pairs = set()
    for i, neighbors in inst.nooverlaps.items():
        for j in neighbors:
            if inst.activity_to_project[i] == inst.activity_to_project[j]:
                u, v = sorted((i, j))
                if u == v or (u, v) in added_pairs:
                    continue
                
                # NoOverlap(I_i, I_j)
                model.AddNoOverlap([intervals[u], intervals[v]])
                added_pairs.add((u, v))

    # Eq 8: Release Times
    for i in range(inst.num_activities):
        p_id = inst.activity_to_project[i]
        rt_p = inst.release_times[p_id]
        model.Add(starts[i] >= rt_p)

    # Eq 9: Tardiness Definition
    # T_p >= End(I_i) - DD_p for all i in project p
    for i in range(inst.num_activities):
        p_id = inst.activity_to_project[i]
        dd_p = inst.due_dates[p_id]
        # Tardy >= End - DueDate
        model.Add(tardy_vars[p_id] >= ends[i] - dd_p)
        
    # Implicit T_p >= 0 is handled by domain definition NewIntVar(0, horizon)

    # --- Objective ---
    model.Minimize(sum(tardy_vars.values()))

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    
    status = solver.Solve(model)

    # --- Result Retrieval ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        obj_val = solver.ObjectiveValue()
        if status == cp_model.OPTIMAL:
            sol_status = "optimal"
            print(f"Optimal solution found! Objective: {obj_val} Time: {solver.WallTime():.2f}s")
        else:
            sol_status = "feasible"
            print(f"Feasible solution found! Objective: {obj_val} Time: {solver.WallTime():.2f}s")

        # Recover Start Times
        start_times = {}
        for i in range(inst.num_activities):
            start_times[i] = solver.Value(starts[i])

        # Recover Assigned Teams
        assigned_teams = {}
        for (i, t), var in presences.items():
            if solver.Value(var) == 1:
                assigned_teams[i] = t

        return obj_val, start_times, assigned_teams, sol_status

    elif status == cp_model.UNKNOWN: # Time limit reached without solution
        print("Time limit reached with no feasible solution.")
        return None
    else:
        print("Model is infeasible.")
        return None
    

def visualize_project_gantt(inst: Instance, start_times: Dict[int, float], assigned_teams: Dict[int, int], limit_x=None):
    num_projects = inst.num_projects
    
    fig, axes = plt.subplots(nrows=num_projects, ncols=1, figsize=(14, 10 * num_projects), sharex=True)
    if num_projects == 1: axes = [axes]

    set_colors = ['#E88C99', '#8fbcd4', "#b1b1b1", '#C1E4A0', '#9EAFCB', "#b1b1b1"]
    pred_map = inst.precedences
    local_max_time = 0

    for p_id in range(num_projects):
        ax = axes[p_id]
        tasks = []
        project_activities = inst.project_to_activities[p_id]
        
        for act_id in project_activities:
            start = start_times.get(act_id, 0)
            duration = inst.durations[act_id]
            end = start + duration
            local_max_time = max(local_max_time, end)
            
            rel_idx = act_id % inst.num_standard_activities

            tasks.append({
                'id': act_id,
                'set_id': inst.activity_to_set[act_id],
                'start': start,
                'end': end,
                'duration': duration,
                'team': assigned_teams.get(act_id, "?"),
                'y': len(project_activities) - 1 - rel_idx, 
                "label": f"{p_id+1}.{rel_idx+1}"
            })
            
        tasks.sort(key=lambda x: x['id'], reverse=True)
        pos_map = {}

        # Grid config
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        ax.set_ylim(-1, len(project_activities))
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.grid(True, which='both', color='#CCCCCC', linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Draw Bars
        for t in tasks:
            y_pos = t['y'] + 0.5
            color = set_colors[t['set_id'] % len(set_colors)]
            
            ax.barh(y=y_pos, width=t['duration'], left=t['start'], 
                    height=0.8, color=color, edgecolor='black', alpha=0.9, zorder=3)
            
            ax.text(t['start'] + t['duration']/2, y_pos-0.1, t['label'], 
                    ha='center', va='center', fontsize=8, color='black', zorder=4)
            
            pos_map[t['id']] = {'start': t['start'], 'end': t['end'], 'y': y_pos}

        # Draw Relations (Arrow)
        arrow_args = dict(arrowstyle="->", color="gray", lw=0.8, connectionstyle="arc3,rad=0.1")
        for t in tasks:
            curr_id = t['id']
            preds = pred_map.get(curr_id, [])
            for pred_id in preds:
                if inst.activity_to_project[pred_id] == p_id:
                    if pred_id in pos_map:
                        pred_info = pos_map[pred_id]
                        ax.annotate("", 
                                    xy=(t['start'], t['y']), 
                                    xytext=(pred_info['end'], pred_info['y']),
                                    arrowprops=arrow_args, zorder=5)

        due = inst.due_dates[p_id]
        ax.axvline(x=due, color='red', linestyle='--', linewidth=2, label='Due Date', zorder=2)
        ax.set_ylabel(f"Project {p_id+1}", fontsize=12, fontweight='bold')

    final_xlim = limit_x if limit_x is not None else (local_max_time + 5)
    axes[-1].set_xlim(0, final_xlim)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.1)
    
    plt.savefig("1.project_gantt.png", dpi=150)
    plt.close()


def plot_resource_profile(inst: Instance, start_times: Dict[int, float], assigned_teams: Dict[int, int], limit_x=None):
    colors_hex = ['#8DA0CB', '#FC8D62', '#A6D854', '#E78AC3', '#FFD92F', '#E5C494']
    plt.rcParams['font.family'] = 'sans-serif'
    
    resource_jobs = {}
    
    # Trace back Company ID from assigned Team ID
    for act_id, start_val in start_times.items():
        team_id = assigned_teams.get(act_id)
        if team_id is None: continue
        
        c_id = inst.team_to_company[team_id]
        if c_id not in resource_jobs:
            resource_jobs[c_id] = []
        
        p_id = inst.activity_to_project[act_id]
        rel_idx = act_id % inst.num_standard_activities
        
        s_int = int(round(start_val))
        d_int = int(inst.durations[act_id])
        
        resource_jobs[c_id].append({
            "p": p_id, 
            "a": act_id, 
            "start": s_int, 
            "duration": d_int, 
            "end": s_int + d_int,
            "label": f"{p_id+1}.{rel_idx+1}"
        })

    sorted_companies = sorted(resource_jobs.keys())
    # Exclude dummy/infinite resource company if needed
    if len(sorted_companies) > 0:
        target_companies = sorted_companies[:-1]
    else:
        target_companies = []

    n_res = len(target_companies)
    if n_res == 0:
        print("No resources to plot.")
        return

    # Calculate Y-axis height (Tetris stacking)
    plot_data = [] 
    max_x_limit = 0 
    y_limits_list = []

    for c_id in target_companies:
        jobs = resource_jobs[c_id]
        capacity = len(inst.company_to_teams[c_id])
        
        jobs.sort(key=lambda x: (x['start'], x['p'], x['a']))
        occupied = set()
        local_max_time = 0
        
        for job in jobs:
            local_max_time = max(local_max_time, job['end'])
            y_level = 0
            while True:
                is_clash = False
                for t in range(job['start'], job['end']):
                    if (t, y_level) in occupied:
                        is_clash = True
                        break
                if not is_clash:
                    job['y'] = y_level
                    for t in range(job['start'], job['end']):
                        occupied.add((t, y_level))
                    break
                y_level += 1
        
        max_x_limit = max(max_x_limit, local_max_time)
        max_y_used = max((j['y'] for j in jobs), default=0) + 1
        current_req_height = max(max_y_used, capacity)
        this_ylim = (int(current_req_height) // 5 + 1) * 5
        y_limits_list.append(this_ylim)
        plot_data.append((c_id, jobs, capacity, this_ylim))

    fig, axes = plt.subplots(
        nrows=n_res, 
        ncols=1, 
        figsize=(14, 2 * n_res), 
        sharex=True,
        gridspec_kw={'height_ratios': y_limits_list} 
    )
    if n_res == 1: axes = [axes]

    for idx, (c_id, jobs, capacity, this_ylim) in enumerate(plot_data):
        ax = axes[idx]
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        ax.grid(True, which='both', color='#CCCCCC', linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        for job in jobs:
            rect_color = colors_hex[job['p'] % len(colors_hex)]
            rect = patches.Rectangle(
                (job['start'], job['y']), 
                job['duration'], 
                1.0, 
                facecolor=rect_color, 
                edgecolor='black', 
                linewidth=1.0,
                alpha=0.9
            )
            ax.add_patch(rect)
            
            cx = job['start'] + job['duration'] / 2
            cy = job['y'] + 0.5
            ax.text(cx, cy, job['label'], ha='center', va='center', fontsize=8, color='black')

        ax.axhline(y=capacity, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_ylabel(f"Company {c_id+1}", fontsize=10, fontweight='bold')
        ax.set_ylim(0, this_ylim)
        
        final_xlim = limit_x if limit_x is not None else (local_max_time + 5)
        ax.set_xlim(0, final_xlim)
        ax.tick_params(axis='y', labelsize=9)
    
    # Legend
    legend_patches = []
    unique_projects = sorted(list(set(j['p'] for d in plot_data for j in d[1])))
    for p_id in unique_projects:
        c = colors_hex[p_id % len(colors_hex)]
        legend_patches.append(patches.Patch(facecolor=c, edgecolor='black', label=f'Project {p_id+1}'))
    
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.95, 0.98), ncol=len(unique_projects))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.1)
    
    plt.savefig("2.resource_profile.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    instance = create_random_instance(num_projects=20)
    # result = solve_rcmpsp_gurobi(instance, 300)
    result = solve_rcmpsp_cp(instance, 300)
    viz = True
    
    if result and viz:
        obj_val, start_times, assigned_teams = result

        # Calculate Horizon for Sync
        max_end_time = 0
        for act_id, s_time in start_times.items():
            duration = instance.durations[act_id]
            end = s_time + duration
            if end > max_end_time:
                max_end_time = end

        unified_limit_x = max_end_time + 5
        
        print("\nVisualizing Results...")
        visualize_project_gantt(instance, start_times, assigned_teams, unified_limit_x)
        
        plot_resource_profile(
            instance,
            start_times,
            assigned_teams,
            unified_limit_x
        )
        print("Done. Saved '1.project_gantt.png' and '2.resource_profile.png'.")