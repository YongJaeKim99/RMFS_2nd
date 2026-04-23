"""
RMFS MILP Solver (Gurobi).

solve_rmfs_milp(data, time_limit) 함수를 통해 외부에서 호출 가능.
단독 실행 시 data.json 파일을 읽어서 풀이.
"""
import json
import time as time_module

try:
    from gurobipy import Model, GRB, quicksum
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False


def solve_rmfs_milp(data, time_limit=600, mip_gap=0.05, verbose=True,
                    extract_solution=False):
    """
    RMFS MILP 모델을 Gurobi로 풀이한다.

    Args:
        data: dict (1-indexed) with keys:
            N_T, P, K, S, R, TT, UT, n_Seq_P, Seq_P, PT, ST, S_P, IS
        time_limit: 시간 제한 (초, default 600)
        mip_gap: MIP Gap 허용치 (default 0.05)
        verbose: Gurobi 출력 표시 여부
        extract_solution: True이면 솔루션 상세 정보도 반환

    Returns:
        extract_solution=False:
            (makespan, status, runtime) or None
        extract_solution=True:
            (makespan, status, runtime, solution_detail) or None
            solution_detail: dict with keys:
                'pod_departures': list of (pod, ws_loc, storage, time)
                    - pod가 WS에서 storage로 출발하는 이벤트 (1-indexed, 시간순)
                'pod_arrivals_at_ws': list of (pod, ws_loc, from_storage, time)
                    - pod가 storage에서 WS로 도착하는 이벤트 (1-indexed, 시간순)
                'processing_events': list of (ws_loc, seq_pos, time)
                    - WS에서 처리 이벤트 (1-indexed)
                'N_S': int (storage 수, 인덱스 변환용)
    """
    if not HAS_GUROBI:
        print("Gurobi가 설치되지 않았습니다.")
        return None

    # Parameters
    N_T = data['N_T']
    P = data['P']
    K = data['K']
    S = data['S']
    L = S + K
    R = data['R']
    TT = data['TT']
    UT = data['UT']
    n_Seq_P = data['n_Seq_P']
    Seq_P = data['Seq_P']
    PT = data['PT']
    ST = data['ST']
    IS = data['IS']
    S_P = data['S_P']

    # Sets
    T = range(1, N_T + 1)
    SP = range(1, n_Seq_P + 1)

    # Model
    m = Model("RMFS_MILP")
    m.setParam('Seed', 42)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', mip_gap)
    m.setParam('OutputFlag', 1 if verbose else 0)

    # Decision variables
    y = m.addVars(K, SP, T, vtype=GRB.BINARY, name="y")
    v = m.addVars(P, L, T, vtype=GRB.BINARY, name="v")
    a = m.addVars(P, L, L, T, vtype=GRB.BINARY, name="a")
    d = m.addVars(P, L, L, T, vtype=GRB.BINARY, name="d")
    lt = m.addVars(R, P, S, T, vtype=GRB.BINARY, name="lt")
    ut = m.addVars(R, P, S, T, vtype=GRB.BINARY, name="ut")
    x = m.addVars(R, P, T, vtype=GRB.BINARY, name="x")
    z = m.addVars(R, S, S, T, T, vtype=GRB.BINARY, name="z")
    MS = m.addVar(vtype=GRB.CONTINUOUS, name="MS")

    # Objective
    m.setObjective(MS, GRB.MINIMIZE)

    # Constraints

    # ct1: Makespan 계산
    m.addConstrs(
        (MS >= (t - 1) * UT * a[p, s, l, t]
         for p in P for s in S for l in L for t in T),
        name="makespan"
    )

    # ct2: 피킹 시간 준수
    m.addConstrs(
        (quicksum(y[k, sp, t] for t in T) == PT[k][sp] // UT
         for k in K for sp in SP if Seq_P[k][sp] != 0),
        name="picking_time"
    )

    # ct33: 설정 시간 준수
    m.addConstrs(
        (y[k, sp, t] + y[k, sp + 1, t + 1] <= 1
         for k in K for sp in SP for t in T
         if sp != n_Seq_P and t != N_T and Seq_P[k][sp] != 0),
        name="setup_time"
    )

    # ct3: 작업장 내 포드 순서
    m.addConstrs(
        (y[k, sp1, t1] + y[k, sp2, t2] <= 1
         for k in K for sp1 in SP for sp2 in SP for t1 in T for t2 in T
         if t2 < t1 and sp1 < sp2),
        name="pod_order"
    )

    # ct4: 작업장에서 동시에 하나의 작업만
    m.addConstrs(
        (quicksum(y[k, sp, t] for sp in SP) <= 1
         for k in K for t in T),
        name="single_task"
    )

    # ct5: 피킹 작업되는 포드는 작업장에 존재
    m.addConstrs(
        (y[k, sp, t] <= v[Seq_P[k][sp], k, t]
         for k in K for sp in SP for t in T if Seq_P[k][sp] != 0),
        name="pod_at_workstation"
    )

    # ct6: 포드 출발 조건
    m.addConstrs(
        (quicksum(d[p, l1, l2, t] for l2 in L if l1 != l2) <= v[p, l1, t - 1]
         for p in P for l1 in L for t in T if t != 1),
        name="departure_condition"
    )

    # ct7: 포드 출발 시 존재하지 않음
    m.addConstrs(
        (quicksum(d[p, l1, l2, t] for l2 in L if l1 != l2) + v[p, l1, t] <= 1
         for p in P for l1 in L for t in T),
        name="departure_exclusive"
    )

    # ct8: 포드 도착 시 존재
    m.addConstrs(
        (quicksum(a[p, l1, l2, t] for l2 in L if l1 != l2) <= v[p, l1, t]
         for p in P for l1 in L for t in T),
        name="arrival_condition"
    )

    # ct9: 포드 도착과 기존 존재 배타적
    m.addConstrs(
        (quicksum(a[p, l1, l2, t] for l2 in L if l1 != l2) + v[p, l1, t - 1] <= 1
         for p in P for l1 in L for t in T if t != 1),
        name="arrival_exclusive"
    )

    # ct10: 저장소 공간 제약
    m.addConstrs(
        (quicksum(v[p, s, t] for p in P) <= 1
         for s in S for t in T),
        name="storage_capacity"
    )

    # ct11: 포드 위치 균형 방정식
    m.addConstrs(
        (v[p, l1, t] == v[p, l1, t - 1] -
         quicksum(d[p, l1, l2, t] for l2 in L if l1 != l2) +
         quicksum(a[p, l1, l2, t] for l2 in L if l1 != l2)
         for p in P for l1 in L for t in T if t != 1),
        name="pod_balance"
    )

    # ct12: 포드는 동시에 도착과 출발 불가
    m.addConstrs(
        (quicksum(d[p, l1, l2, t] for l2 in L if l1 != l2) +
         quicksum(a[p, l1, l2, t] for l2 in L if l1 != l2) <= 1
         for p in P for l1 in L for t in T),
        name="no_simultaneous_move"
    )

    # ct13: 포드 초기 위치 제약
    m.addConstrs(
        (v[p, l, 1] <= 0
         for p in P for l in L if l != S_P[p]),
        name="initial_position"
    )

    # ct14: 포드 초기 출발 제약
    m.addConstrs(
        (d[p, l1, l2, 1] <= 0
         for p in P for l1 in L for l2 in L if l1 != S_P[p]),
        name="initial_departure"
    )

    # ct15, ct16: 자기 위치로 이동 금지
    m.addConstrs(
        (d[p, l, l, t] <= 0
         for p in P for l in L for t in T),
        name="no_self_departure"
    )
    m.addConstrs(
        (a[p, l, l, t] <= 0
         for p in P for l in L for t in T),
        name="no_self_arrival"
    )

    # ct17: 출발 후 도착 시간 관계
    for p in P:
        for l1 in L:
            for l2 in L:
                for t1 in T:
                    if l1 != l2:
                        travel_time = TT[l1][l2] // UT
                        t2 = t1 + travel_time
                        if t2 in T:
                            m.addConstr(
                                d[p, l1, l2, t1] == a[p, l2, l1, t2],
                                name=f"departure_arrival_{p}_{l1}_{l2}_{t1}_{t2}"
                            )

    # ct18: 포드 최소 도착 시간
    m.addConstrs(
        (a[p, l1, l2, t] <= 0
         for p in P for l1 in L for l2 in L for t in T
         if l1 != l2 and t <= TT[l1][l2] // UT),
        name="min_arrival_time"
    )

    # ct19: 마지막에 저장소로 복귀
    m.addConstrs(
        (quicksum(v[p, s, N_T] for s in S) == 1
         for p in P),
        name="final_storage"
    )

    # ct20, ct21: 저장소 간 직접 이동 금지
    m.addConstrs(
        (d[p, s1, s2, t] <= 0
         for p in P for s1 in S for s2 in S for t in T),
        name="no_storage_departure"
    )
    m.addConstrs(
        (a[p, s1, s2, t] <= 0
         for p in P for s1 in S for s2 in S for t in T),
        name="no_storage_arrival"
    )

    # 로봇 관련 제약조건

    # ct22: 포드는 하나의 로봇에 의해서만 점유
    m.addConstrs(
        (quicksum(x[r, p, t] for r in R) <= 1
         for p in P for t in T),
        name="single_robot_per_pod"
    )

    # ct23: 로봇은 하나의 포드만 점유
    m.addConstrs(
        (quicksum(x[r, p, t] for p in P) <= 1
         for r in R for t in T),
        name="single_pod_per_robot"
    )

    # ct24: 로봇 점유 균형 방정식
    m.addConstrs(
        (x[r, p, t] == x[r, p, t - 1] -
         quicksum(ut[r, p, s, t] for s in S) +
         quicksum(lt[r, p, s, t] for s in S)
         for p in P for r in R for t in T if t != 1),
        name="robot_balance"
    )

    # ct25: 로봇 점유 초기 조건
    for p in P:
        for r in R:
            if IS[r] == S_P[p]:
                m.addConstr(
                    x[r, p, 1] == quicksum(lt[r, p, s, 1] for s in S),
                    name=f"robot_initial_{p}_{r}"
                )

    # ct26: 로봇은 동시에 로드와 언로드 불가
    m.addConstrs(
        (quicksum(ut[r, p, s, t] for s in S for p in P) +
         quicksum(lt[r, p, s, t] for s in S for p in P) <= 1
         for r in R for t in T),
        name="no_simultaneous_load_unload"
    )

    # ct254: 저장소 지연 제약
    m.addConstrs(
        (quicksum(ut[r, p, s, t] for r in R for p in P) +
         quicksum(lt[r, p, s, t] for r in R for p in P) <= 1
         for s in S for t in T),
        name="storage_delay"
    )

    # ct27: 포드 출발 시 로봇 로드
    m.addConstrs(
        (quicksum(d[p, s, k, t] for k in K) ==
         quicksum(lt[r, p, s, t] for r in R)
         for p in P for s in S for t in T if t != N_T),
        name="departure_load"
    )

    # ct28: 포드 도착 시 로봇 언로드
    m.addConstrs(
        (quicksum(a[p, s, k, t] for k in K) ==
         quicksum(ut[r, p, s, t] for r in R)
         for p in P for s in S for t in T if t != 1),
        name="arrival_unload"
    )

    # ct29: 언로드 조건
    m.addConstrs(
        (ut[r, p, s, t] <= x[r, p, t - 1]
         for p in P for s in S for r in R for t in T if t != 1),
        name="unload_condition"
    )

    # ct30: 로봇 이동 시간 제약
    m.addConstrs(
        (ut[r, p1, s1, t1] + lt[r, p2, s2, t2] <= 1
         for p1 in P for p2 in P for r in R
         for s1 in S for s2 in S for t1 in T for t2 in T
         if p1 != p2 and t2 >= t1 and t2 < t1 + TT[s1][s2] // UT),
        name="robot_travel_time"
    )

    # ct31: 로봇 초기 이동 시간
    m.addConstrs(
        (lt[r, p, s, t] <= 0
         for p in P for r in R for s in S for t in T
         if t <= TT[IS[r]][s] // UT),
        name="robot_initial_travel"
    )

    # 최적화 실행
    if verbose:
        print("=== RMFS MILP 최적화 시작 ===")
        print(f"포드 수: {len(P)}, 작업장 수: {len(K)}, "
              f"저장소 수: {len(S)}, 로봇 수: {len(R)}, 시간 구간: {N_T}")

    m.optimize()

    # 솔루션 상세 추출 함수
    def _extract_solution_detail():
        """Gurobi 변수에서 pod-storage 배정 정보를 추출한다."""
        N_S_count = len(S)

        # 1) Pod departures: WS → Storage
        pod_departures = []
        for p in P:
            for k in K:
                for s in S:
                    for t in T:
                        if d[p, k, s, t].X > 0.5:
                            pod_departures.append((p, k, s, t))

        # 2) Pod arrivals at WS: Storage → WS
        pod_arrivals_at_ws = []
        for p in P:
            for k in K:
                for s in S:
                    for t in T:
                        if a[p, k, s, t].X > 0.5:
                            pod_arrivals_at_ws.append((p, k, s, t))

        # 3) Processing events
        processing_events = []
        for k in K:
            for sp in SP:
                if Seq_P[k][sp] != 0:
                    for t in T:
                        if y[k, sp, t].X > 0.5:
                            processing_events.append((k, sp, t))

        # 시간순 정렬
        pod_departures.sort(key=lambda x: x[3])
        pod_arrivals_at_ws.sort(key=lambda x: x[3])
        processing_events.sort(key=lambda x: x[2])

        return {
            'pod_departures': pod_departures,
            'pod_arrivals_at_ws': pod_arrivals_at_ws,
            'processing_events': processing_events,
            'N_S': N_S_count,
        }

    # 결과 반환
    def _make_result(makespan_val, status_str, runtime_val):
        if extract_solution:
            sol_detail = _extract_solution_detail()
            return (makespan_val, status_str, runtime_val, sol_detail)
        return (makespan_val, status_str, runtime_val)

    if m.status == GRB.OPTIMAL:
        if verbose:
            print(f"최적해 발견: Makespan = {m.ObjVal:.4f}")
        return _make_result(m.ObjVal, 'optimal', m.Runtime)

    elif m.status == GRB.TIME_LIMIT:
        if m.solCount > 0:
            if verbose:
                print(f"시간 제한 내 최선해: Makespan = {m.ObjVal:.4f}, "
                      f"Gap = {m.MIPGap * 100:.2f}%")
            return _make_result(m.ObjVal, 'time_limit', m.Runtime)
        else:
            if verbose:
                print("시간 제한 내 해를 찾지 못함")
            return None

    elif m.status == GRB.INFEASIBLE:
        if verbose:
            print("모델이 실행 불가능 (infeasible)")
        return None

    else:
        if m.solCount > 0:
            if verbose:
                print(f"Gurobi status {m.status}, 해 존재: Makespan = {m.ObjVal:.4f}")
            return _make_result(m.ObjVal, 'feasible', m.Runtime)
        if verbose:
            print(f"Gurobi status {m.status}, 해 없음")
        return None


# =================================================================
# 단독 실행: data.json 파일에서 읽어서 풀기
# =================================================================
if __name__ == "__main__":
    import sys

    json_path = 'MILP/data.json'
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    with open(json_path, 'r') as f:
        data = json.load(f)

    # JSON 문자열 키를 정수로 변환
    for key in ['TT', 'Seq_P', 'PT']:
        if key in data:
            converted = {}
            for k in data[key]:
                converted[int(k)] = {}
                for sub_k in data[key][k]:
                    converted[int(k)][int(sub_k)] = data[key][k][sub_k]
            data[key] = converted

    for key in ['S_P', 'IS']:
        if key in data:
            converted = {}
            for k in data[key]:
                converted[int(k)] = data[key][k]
            data[key] = converted

    result = solve_rmfs_milp(data, time_limit=600, verbose=True)
    if result:
        makespan, status, runtime = result
        print(f"\n=== 결과 ===")
        print(f"Makespan: {makespan:.4f}")
        print(f"Status: {status}")
        print(f"Runtime: {runtime:.2f}s")
    else:
        print("\n=== 해를 찾지 못함 ===")
