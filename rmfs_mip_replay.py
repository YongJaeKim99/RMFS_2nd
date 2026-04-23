"""
MIP Replay Script for RMFS.

MIP(Gurobi) 솔루션을 RMFS 환경에서 step-by-step replay하여
MIP 솔루션의 재현 정확도를 검증한다.

사용법:
  KMP_DUPLICATE_LIB_OK=TRUE "C:\\Users\\YongJae\\anaconda3\\envs\\RSS_1st\\python.exe" rmfs_mip_replay.py

데이터 흐름:
  1) 테스트 데이터 pickle 로드 (seeds, env_params)
  2) 각 인스턴스마다 MIP 풀이 + 솔루션 추출
  3) 동일 seed로 RMFS 환경을 reset하고 MIP 솔루션 기반 action 선택
  4) MIP makespan vs Replay makespan 비교
  5) 결과를 Excel + CSV로 저장
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import io
import time
import csv
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def build_mip_action_map(solution_detail, env):
    """
    MIP 솔루션에서 (pod, ws, visit_number) -> storage action 매핑을 구축한다.

    Args:
        solution_detail: MILP.solve_rmfs_milp(extract_solution=True)의 반환값
        env: reset 직후의 RMFS_Environment (Pod_Sequence_in_WS 등 참조용)

    Returns:
        action_map: dict[(pod_0idx, ws_0idx, visit_num)] -> action (0=stay, 1..N_S)
            visit_num은 해당 pod가 해당 WS를 방문하는 순서 (0-based)
    """
    N_S = solution_detail['N_S']
    pod_departures = solution_detail['pod_departures']

    # WS별 pod의 방문 순서를 추적하기 위해 departure를 시간순으로 그룹화
    # pod_departures: list of (pod_1idx, ws_loc_1idx, storage_1idx, time)
    # ws_loc_1idx = N_S + ws_0idx + 1  →  ws_0idx = ws_loc_1idx - N_S - 1
    # pod_1idx → pod_0idx = pod_1idx - 1
    # storage_1idx → action = storage_1idx (action i = storage i-1 in 0-indexed)

    # (pod_0idx, ws_0idx) -> list of (time, storage_action) 정렬
    departure_by_pod_ws = defaultdict(list)
    for pod_1, ws_loc_1, storage_1, t in pod_departures:
        pod_0 = pod_1 - 1
        ws_0 = ws_loc_1 - N_S - 1
        action = storage_1  # action i (1..N_S) = storage i-1 (0-indexed)
        departure_by_pod_ws[(pod_0, ws_0)].append((t, action))

    # 시간순 정렬 후 visit number 부여
    action_map = {}
    for (pod_0, ws_0), departures in departure_by_pod_ws.items():
        departures.sort(key=lambda x: x[0])
        for visit_num, (t, action) in enumerate(departures):
            action_map[(pod_0, ws_0, visit_num)] = action

    return action_map


def replay_mip_solution(env, action_map, seed, N_P, N_R, Total_PodTask, N_W,
                        verbose=False):
    """
    MIP 솔루션을 RMFS 환경에서 replay한다.

    Args:
        env: RMFS_Environment 인스턴스
        action_map: build_mip_action_map()의 반환값
        seed, N_P, N_R, Total_PodTask, N_W: env.reset() 파라미터
        verbose: 디버그 출력 여부

    Returns:
        diagnostics: dict with keys:
            'done': bool - 정상 완료 여부
            'replay_makespan': float - replay 결과 makespan
            'step_count': int - 총 step 수
            'match_steps': list - MIP 매칭 성공 step 정보
            'fallback_steps': list - MIP에 없어서 fallback한 step 정보
            'infeasible_steps': list - action이 infeasible했던 step 정보
            'decision_log': list of (step, pod, ws, action, note)
    """
    # 환경 reset (print 억제)
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        env.reset(seed, N_P, N_R, Total_PodTask, N_W)
    finally:
        sys.stdout = _stdout

    # (pod, ws) 별 방문 카운터
    visit_counter = defaultdict(int)

    match_steps = []
    fallback_steps = []
    infeasible_steps = []
    decision_log = []

    step_count = 0
    done = False

    while not done:
        curpod = env.curpod
        curws = env.curws
        action_mask = env.graph_state['action_mask']

        # 현재 (pod, ws) 방문 번호
        visit_num = visit_counter[(curpod, curws)]

        # MIP 솔루션에서 action 조회
        mip_key = (curpod, curws, visit_num)
        mip_action = action_map.get(mip_key, None)

        note = ''

        if mip_action is not None:
            # MIP 솔루션에 해당하는 action이 있음
            if action_mask[mip_action]:
                # MIP action이 feasible
                action = mip_action
                note = 'MIP_MATCH'
                match_steps.append((step_count, curpod, curws, action))
            else:
                # MIP action이 현재 env에서 infeasible → fallback
                note = f'MIP_INFEASIBLE(mip_act={mip_action})'
                infeasible_steps.append(
                    (step_count, curpod, curws, mip_action, 'action_masked'))

                # Fallback: action_mask에서 첫 번째 가능한 action 선택
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = int(valid_actions[0])
                    note += f'->fallback({action})'
                else:
                    action = 0
                    note += '->forced(0)'
        else:
            # MIP 솔루션에 매핑 없음 → pod가 WS에 머무르는 경우 (action=0)
            # 또는 MIP이 해당 departure를 포함하지 않은 경우
            note = 'MIP_NO_MATCH'

            # Pod가 더 이상 다른 WS에서 task가 없으면 storage로 반납해야 함
            # → action_mask에서 feasible한 action 중 선택
            if action_mask[0]:
                action = 0  # Stay가 가능하면 Stay
                note += '->stay(0)'
            else:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = int(valid_actions[0])
                    note += f'->fallback({action})'
                else:
                    action = 0
                    note += '->forced(0)'

            fallback_steps.append((step_count, curpod, curws, action, note))

        decision_log.append((step_count, curpod, curws, action, note))

        if verbose:
            print(f"  step {step_count}: pod={curpod}, ws={curws}, "
                  f"visit={visit_num}, action={action} [{note}]")

        # 방문 카운터 증가
        visit_counter[(curpod, curws)] += 1

        # 환경 step
        _, reward, done, infeasible, makespan = env.step(action)

        if infeasible:
            if verbose:
                print(f"  step {step_count}: INFEASIBLE! action={action}")
            break

        step_count += 1

    replay_makespan = env.Makespan if not infeasible else None

    return {
        'done': done and not infeasible,
        'replay_makespan': replay_makespan,
        'step_count': step_count,
        'match_steps': match_steps,
        'fallback_steps': fallback_steps,
        'infeasible_steps': infeasible_steps,
        'decision_log': decision_log,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent

    # =================================================================
    # Configuration
    # =================================================================

    # 테스트 데이터 경로
    TEST_DATA_DIR = 'data/rmfs_test'

    # MIP 설정
    MIP_TIME_LIMIT = 300      # 인스턴스당 시간 제한 (초)
    MIP_GAP = 0.05            # MIP Gap 허용치

    # 인스턴스 범위 필터링 (None이면 전체)
    INSTANCE_START = 0
    INSTANCE_END = None

    # 출력 옵션
    SAVE_REPLAY_DECISIONS_CSV = True
    VERBOSE_REPLAY = False

    # 디바이스
    DEVICE_MODE = 'cpu'

    # =================================================================
    # 설정 출력
    # =================================================================
    print("=" * 60)
    print("MIP Replay for RMFS")
    print("=" * 60)
    print(f"  TEST_DATA_DIR:        {TEST_DATA_DIR}")
    print(f"  MIP_TIME_LIMIT:       {MIP_TIME_LIMIT}s")
    print(f"  MIP_GAP:              {MIP_GAP}")
    print(f"  INSTANCE_RANGE:       [{INSTANCE_START}, {INSTANCE_END})")
    print(f"  SAVE_DECISIONS_CSV:   {SAVE_REPLAY_DECISIONS_CSV}")
    print(f"  VERBOSE_REPLAY:       {VERBOSE_REPLAY}")
    print("=" * 60)

    # =================================================================
    # 테스트 데이터 로드
    # =================================================================
    data_base_dir = project_root / TEST_DATA_DIR

    if not data_base_dir.exists():
        print(f"테스트 데이터 폴더가 존재하지 않습니다: {data_base_dir}")
        print(f"rmfs_data_generation.py를 먼저 실행해주세요.")
        sys.exit(1)

    pickle_files = sorted(data_base_dir.glob("*.pickle"))
    if not pickle_files:
        print(f"{data_base_dir}에 .pickle 파일이 없습니다.")
        sys.exit(1)

    print(f"\n  테스트 데이터 파일 {len(pickle_files)}개 발견")
    for pf in pickle_files:
        print(f"    - {pf.name}")

    # =================================================================
    # Import
    # =================================================================
    from RMFS_ENV import RMFS_Environment
    from rmfs_milp_utils import convert_env_to_milp_data
    from MILP import solve_rmfs_milp, HAS_GUROBI

    if not HAS_GUROBI:
        print("Gurobi가 설치되지 않았습니다. MIP Replay를 실행할 수 없습니다.")
        sys.exit(1)

    # =================================================================
    # 결과 저장 폴더
    # =================================================================
    results_dir = project_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = results_dir / f"mip_replay_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_REPLAY_DECISIONS_CSV:
        decisions_dir = session_dir / "replay_decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 처리
    # =================================================================
    all_results = []
    total_start_time = time.time()

    for pickle_path in pickle_files:
        pickle_name = pickle_path.stem
        print(f"\n{'=' * 60}")
        print(f"  Processing: {pickle_path.name}")
        print(f"{'=' * 60}")

        with open(pickle_path, 'rb') as f:
            problem = pickle.load(f)

        seeds = problem['seeds']
        N_P = problem['N_P']
        N_R = problem['N_R']
        N_W = problem['N_W']
        Total_PodTask = problem['Total_PodTask']
        saved_env_params = problem.get('env_params', {})

        total_instances = len(seeds)
        inst_start = INSTANCE_START if INSTANCE_START is not None else 0
        inst_end = INSTANCE_END if INSTANCE_END is not None else total_instances
        inst_start = max(0, min(inst_start, total_instances))
        inst_end = max(inst_start, min(inst_end, total_instances))
        num_instances = inst_end - inst_start

        print(f"  Seeds: {total_instances}개, "
              f"실행 범위: [{inst_start}, {inst_end}) ({num_instances}개)")

        # 환경 파라미터
        block_rows = saved_env_params.get('block_rows', 3)
        block_cols = saved_env_params.get('block_cols', 3)
        block_h = saved_env_params.get('block_h', 4)
        block_w = saved_env_params.get('block_w', 2)
        Unit_PT = saved_env_params.get('Unit_PT', 15)
        ST = saved_env_params.get('ST', 5)
        UT = saved_env_params.get('UT', 1)
        Large = saved_env_params.get('Large', True)
        force_mask_stay = saved_env_params.get('force_mask_stay', True)

        # RMFS 환경 생성 (print 억제)
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            env = RMFS_Environment(
                block_rows, block_cols, block_h, block_w,
                Unit_PT, ST, UT, Large,
                force_mask_stay=force_mask_stay)
        finally:
            sys.stdout = _stdout

        file_results = []

        for i in range(inst_start, inst_end):
            print(f"\n   Instance {i}")
            instance_start_time = time.time()

            try:
                # 1. MIP 풀기 (솔루션 추출 포함)
                # 환경 reset (MILP 데이터 변환용)
                _stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    env.reset(seeds[i], N_P, N_R, Total_PodTask, N_W)
                finally:
                    sys.stdout = _stdout

                milp_data = convert_env_to_milp_data(env)
                print(f"     MILP 데이터: P={len(milp_data['P'])}, "
                      f"K={len(milp_data['K'])}, "
                      f"S={len(milp_data['S'])}, "
                      f"R={len(milp_data['R'])}, "
                      f"N_T={milp_data['N_T']}")

                mip_start = time.time()
                result = solve_rmfs_milp(
                    milp_data,
                    time_limit=MIP_TIME_LIMIT,
                    mip_gap=MIP_GAP,
                    verbose=False,
                    extract_solution=True)
                mip_runtime = time.time() - mip_start

                if result is None:
                    print(f"     [MIP] 해를 찾지 못함 ({mip_runtime:.1f}s)")
                    file_results.append({
                        'instance': i,
                        'mip_makespan': None,
                        'mip_status': 'no_solution',
                        'replay_makespan': None,
                        'diff': None,
                        'match': 'MIP_FAIL',
                        'match_steps': 0,
                        'fallback_steps': 0,
                        'infeasible_steps': 0,
                        'mip_runtime': mip_runtime,
                        'replay_runtime': 0,
                    })
                    continue

                mip_makespan, mip_status, gurobi_runtime, solution_detail = result
                print(f"     [MIP] Makespan: {mip_makespan:.4f}, "
                      f"Status: {mip_status}, Runtime: {mip_runtime:.1f}s")
                print(f"     [MIP] Departures: {len(solution_detail['pod_departures'])}, "
                      f"Arrivals: {len(solution_detail['pod_arrivals_at_ws'])}")

                # 2. Action map 구축
                action_map = build_mip_action_map(solution_detail, env)
                print(f"     [MIP] Action map entries: {len(action_map)}")

                # 3. Replay
                replay_start = time.time()
                diagnostics = replay_mip_solution(
                    env, action_map, seeds[i],
                    N_P, N_R, Total_PodTask, N_W,
                    verbose=VERBOSE_REPLAY)
                replay_runtime = time.time() - replay_start

                replay_makespan = diagnostics['replay_makespan']
                n_match = len(diagnostics['match_steps'])
                n_fallback = len(diagnostics['fallback_steps'])
                n_infeasible = len(diagnostics['infeasible_steps'])

                if diagnostics['done'] and replay_makespan is not None:
                    diff = abs(replay_makespan - mip_makespan)
                    has_issue = n_fallback > 0 or n_infeasible > 0
                    match_str = "DIFF" if has_issue else "MATCH"

                    print(f"     [REPLAY] Makespan: {replay_makespan:.4f}")
                    print(f"     [REPLAY] Diff: {diff:.4f} ({match_str})")
                    print(f"     [REPLAY] Steps: {diagnostics['step_count']}, "
                          f"Match: {n_match}, Fallback: {n_fallback}, "
                          f"Infeasible: {n_infeasible}")

                    if n_infeasible > 0:
                        print(f"     Infeasible steps (최대 5개):")
                        for s, pod, ws, act, reason in diagnostics['infeasible_steps'][:5]:
                            print(f"        step {s}: pod={pod}, ws={ws}, "
                                  f"act={act} ({reason})")

                    if n_fallback > 0:
                        print(f"     Fallback steps (최대 5개):")
                        for s, pod, ws, act, note in diagnostics['fallback_steps'][:5]:
                            print(f"        step {s}: pod={pod}, ws={ws}, "
                                  f"act={act} ({note})")

                    # Replay decision CSV 저장
                    if SAVE_REPLAY_DECISIONS_CSV:
                        try:
                            csv_path = (decisions_dir /
                                        f"{pickle_name}_instance_{i}.csv")
                            with open(csv_path, 'w', newline='') as rf:
                                rf.write(
                                    f"# pickle={pickle_name} instance={i} "
                                    f"mip_makespan={mip_makespan} "
                                    f"replay_makespan={replay_makespan} "
                                    f"match={match_str}\n")
                                writer = csv.writer(rf)
                                writer.writerow([
                                    'step', 'pod', 'ws', 'action', 'note'])
                                for step_n, pod, ws, act, note in \
                                        diagnostics['decision_log']:
                                    writer.writerow(
                                        [step_n, pod, ws, act, note])
                            print(f"     Replay decisions saved: "
                                  f"{csv_path.name}")
                        except Exception as e:
                            print(f"     CSV 저장 실패: {e}")

                    file_results.append({
                        'instance': i,
                        'mip_makespan': mip_makespan,
                        'mip_status': mip_status,
                        'replay_makespan': replay_makespan,
                        'diff': diff,
                        'match': match_str,
                        'match_steps': n_match,
                        'fallback_steps': n_fallback,
                        'infeasible_steps': n_infeasible,
                        'mip_runtime': mip_runtime,
                        'replay_runtime': replay_runtime,
                    })
                else:
                    print(f"     [REPLAY] 미완료 "
                          f"(steps={diagnostics['step_count']})")
                    file_results.append({
                        'instance': i,
                        'mip_makespan': mip_makespan,
                        'mip_status': mip_status,
                        'replay_makespan': None,
                        'diff': None,
                        'match': 'INCOMPLETE',
                        'match_steps': n_match,
                        'fallback_steps': n_fallback,
                        'infeasible_steps': n_infeasible,
                        'mip_runtime': mip_runtime,
                        'replay_runtime': replay_runtime,
                    })

            except Exception as e:
                print(f"     오류: {e}")
                import traceback
                traceback.print_exc()
                total_runtime = time.time() - instance_start_time
                file_results.append({
                    'instance': i,
                    'mip_makespan': None,
                    'mip_status': 'error',
                    'replay_makespan': None,
                    'diff': None,
                    'match': 'ERROR',
                    'match_steps': 0,
                    'fallback_steps': 0,
                    'infeasible_steps': 0,
                    'mip_runtime': total_runtime,
                    'replay_runtime': 0,
                })

        # 파일별 결과 요약
        if file_results:
            match_count = sum(1 for r in file_results
                              if r['match'] == 'MATCH')
            diff_count = sum(1 for r in file_results
                             if r['match'] == 'DIFF')
            mip_objs = [r['mip_makespan'] for r in file_results
                        if r['mip_makespan'] is not None]
            replay_objs = [r['replay_makespan'] for r in file_results
                           if r['replay_makespan'] is not None]

            print(f"\n  MIP_REPLAY 결과 요약 ({pickle_name}):")
            print(f"    MATCH: {match_count}/{len(file_results)}")
            print(f"    DIFF:  {diff_count}/{len(file_results)}")
            if mip_objs:
                print(f"    평균 MIP makespan:    {np.mean(mip_objs):.4f}")
            if replay_objs:
                print(f"    평균 Replay makespan: {np.mean(replay_objs):.4f}")

            # Excel 저장
            output_path = session_dir / f'MIP_Replay_{pickle_name}.xlsx'
            detailed_data = []
            for r in file_results:
                detailed_data.append({
                    'Instance': r['instance'],
                    'MIP Makespan': r['mip_makespan'],
                    'MIP Status': r['mip_status'],
                    'Replay Makespan': r['replay_makespan'],
                    'Diff': r['diff'],
                    'Match': r['match'],
                    'Match Steps': r['match_steps'],
                    'Fallback Steps': r['fallback_steps'],
                    'Infeasible Steps': r['infeasible_steps'],
                    'MIP Runtime': r['mip_runtime'],
                    'Replay Runtime': r['replay_runtime'],
                })

            exp_rows = [
                {'Item': 'Algorithm', 'Value': 'MIP_REPLAY'},
                {'Item': 'Data File', 'Value': pickle_name},
                {'Item': 'MIP Time Limit', 'Value': MIP_TIME_LIMIT},
                {'Item': 'MIP Gap', 'Value': MIP_GAP},
                {'Item': 'Num Instances', 'Value': len(file_results)},
                {'Item': 'MATCH Count', 'Value': match_count},
                {'Item': 'DIFF Count', 'Value': diff_count},
            ]
            if mip_objs:
                exp_rows.append({
                    'Item': 'Avg MIP Makespan',
                    'Value': f'{np.mean(mip_objs):.4f}'})
            if replay_objs:
                exp_rows.append({
                    'Item': 'Avg Replay Makespan',
                    'Value': f'{np.mean(replay_objs):.4f}'})

            exp_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])
            detailed_df = pd.DataFrame(detailed_data)
            num_cols = detailed_df.select_dtypes(include=[np.number]).columns
            detailed_df[num_cols] = detailed_df[num_cols].round(4)

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                exp_df.to_excel(writer, sheet_name='Experiment_Info',
                                index=False)
                detailed_df.to_excel(writer, sheet_name='MIP_Replay_Results',
                                     index=False)

            print(f"  결과가 {output_path}에 저장되었습니다.")

        all_results.extend(file_results)

    # =================================================================
    # 전체 요약
    # =================================================================
    total_time = time.time() - total_start_time

    print(f"\n{'=' * 60}")
    print("MIP Replay 전체 요약")
    print(f"{'=' * 60}")

    if all_results:
        total_match = sum(1 for r in all_results if r['match'] == 'MATCH')
        total_diff = sum(1 for r in all_results if r['match'] == 'DIFF')
        total_incomplete = sum(1 for r in all_results
                               if r['match'] == 'INCOMPLETE')
        total_mip_fail = sum(1 for r in all_results
                              if r['match'] == 'MIP_FAIL')
        total_error = sum(1 for r in all_results if r['match'] == 'ERROR')
        all_mip_objs = [r['mip_makespan'] for r in all_results
                        if r['mip_makespan'] is not None]
        all_replay_objs = [r['replay_makespan'] for r in all_results
                           if r['replay_makespan'] is not None]

        print(f"  총 인스턴스:      {len(all_results)}")
        print(f"  MATCH:            {total_match}")
        print(f"  DIFF:             {total_diff}")
        if total_incomplete > 0:
            print(f"  INCOMPLETE:       {total_incomplete}")
        if total_mip_fail > 0:
            print(f"  MIP_FAIL:         {total_mip_fail}")
        if total_error > 0:
            print(f"  ERROR:            {total_error}")
        if all_mip_objs:
            print(f"  평균 MIP makespan:    {np.mean(all_mip_objs):.4f}")
        if all_replay_objs:
            print(f"  평균 Replay makespan: {np.mean(all_replay_objs):.4f}")
    else:
        print("  결과 없음")

    print(f"  소요 시간:        {total_time:.1f}s")
    print(f"  결과 폴더:        {session_dir}")
    print("\nDone.")
