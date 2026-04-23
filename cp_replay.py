"""
CP Replay Script
CP solution pickle 파일을 읽어 RL 환경에서 replay하여
CP 솔루션의 재현 정확도를 검증한다.

사용법:
  KMP_DUPLICATE_LIB_OK=TRUE "C:\\Users\\YongJae\\anaconda3\\envs\\RSS_1st\\python.exe" cp_replay.py

데이터 흐름:
  입력: data/il/cp_solutions/*.pickle   (CP solution, test.py에서 생성)
      + data/il/il_label_instance/*.pickle (problem 데이터)
  출력: results/ 폴더에 Excel 결과, 선택적 Gantt chart, replay decision CSV
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import pickle
import time
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from il_utils import load_cp_solutions, replay_cp_solution_instance
from gantt_chart import create_gantt_chart_from_schedule

if __name__ == "__main__":
    project_root = Path(__file__).parent

    # =================================================================
    # Configuration
    # =================================================================

    # 입력 폴더
    CP_SOLUTION_DIR = 'data/il/cp_solutions'           # CP solution pickle 폴더
    IL_INSTANCE_DIR = 'data/il/il_label_instance'      # problem pickle 폴더

    # CP solution 파일 지정 (None이면 CP_SOLUTION_DIR 내 전체 파일 처리)
    CP_SOLUTION_FILES = None   # 예: ['test_batch_cp_solution.pickle']

    # Active schedule 변환 사용 여부
    USE_ACTIVE_SCHEDULE = False

    # Action 선택 기준
    ACTION_SELECTION = 'start_time'  # 'start_time', 'end_time', 'reservation'

    # 환경 옵션 (action masking에 영향)
    ALLOW_WAIT_RELEASE = True
    ALLOW_WAIT_MUTEX = True
    ALLOW_WAIT_TEAM = True
    ALLOW_WAIT_PRED = True
    USE_MUTEX_ATTENTION = True

    # 목적함수
    OBJECTIVE = 'tardiness'  # 'tardiness' or 'makespan'

    # 인스턴스 범위 필터링 (None이면 전체)
    INSTANCE_START = 0
    INSTANCE_END = None

    # 출력 옵션
    SHOW_GANTT_CHART = False
    SAVE_REPLAY_DECISIONS_CSV = True

    # 디바이스
    DEVICE_MODE = 'cpu'  # 'cpu' or 'gpu'

    DEBUG_ENV = False

    # =================================================================
    # 디바이스 설정
    # =================================================================
    if DEVICE_MODE == 'gpu' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # =================================================================
    # 설정 출력
    # =================================================================
    print("=" * 60)
    print("CP Replay")
    print("=" * 60)
    print(f"  CP_SOLUTION_DIR:      {CP_SOLUTION_DIR}")
    print(f"  IL_INSTANCE_DIR:      {IL_INSTANCE_DIR}")
    print(f"  USE_ACTIVE_SCHEDULE:  {USE_ACTIVE_SCHEDULE}")
    print(f"  ACTION_SELECTION:     {ACTION_SELECTION}")
    print(f"  OBJECTIVE:            {OBJECTIVE}")
    print(f"  ALLOW_WAIT_RELEASE:   {ALLOW_WAIT_RELEASE}")
    print(f"  ALLOW_WAIT_MUTEX:     {ALLOW_WAIT_MUTEX}")
    print(f"  ALLOW_WAIT_TEAM:      {ALLOW_WAIT_TEAM}")
    print(f"  ALLOW_WAIT_PRED:      {ALLOW_WAIT_PRED}")
    print(f"  USE_MUTEX_ATTENTION:  {USE_MUTEX_ATTENTION}")
    print(f"  INSTANCE_RANGE:       [{INSTANCE_START}, {INSTANCE_END})")
    print(f"  SHOW_GANTT_CHART:     {SHOW_GANTT_CHART}")
    print(f"  SAVE_DECISIONS_CSV:   {SAVE_REPLAY_DECISIONS_CSV}")
    print(f"  DEVICE:               {device}")
    print("=" * 60)

    # =================================================================
    # CP solution 파일 탐색
    # =================================================================
    solution_dir = project_root / CP_SOLUTION_DIR
    if not solution_dir.exists():
        print(f"CP solution 폴더가 존재하지 않습니다: {solution_dir}")
        print(f"  test.py에서 SAVE_CP_SOLUTIONS=True로 먼저 실행해주세요.")
        exit(1)

    if CP_SOLUTION_FILES is not None:
        solution_files = [solution_dir / f for f in CP_SOLUTION_FILES]
    else:
        # _cp_solution.pickle 및 하위 호환 _cp_label.pickle 모두 탐색
        solution_files = sorted(solution_dir.glob("*_cp_solution.pickle"))
        if not solution_files:
            solution_files = sorted(solution_dir.glob("*_cp_label.pickle"))

    if not solution_files:
        print(f"CP solution 파일이 없습니다: {solution_dir}")
        exit(1)

    print(f"\n  CP solution 파일 {len(solution_files)}개 발견")
    for sf in solution_files:
        print(f"    - {sf.name}")

    # =================================================================
    # 결과 저장 폴더 설정
    # =================================================================
    results_dir = project_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = results_dir / f"cp_replay_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    if SHOW_GANTT_CHART:
        gantt_dir = session_dir / "gantt_charts"
        gantt_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_REPLAY_DECISIONS_CSV:
        decisions_dir = session_dir / "replay_decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 처리
    # =================================================================
    all_results = []
    all_active_stats = []
    total_start_time = time.time()

    for solution_path in solution_files:
        print(f"\n{'='*60}")
        print(f"  Processing: {solution_path.name}")
        print(f"{'='*60}")

        # 1. CP solution 로드
        cp_data = load_cp_solutions(solution_path)
        source_pickle_rel = cp_data['source_pickle']
        cp_objective = cp_data.get('objective', OBJECTIVE)
        pickle_name = Path(source_pickle_rel).stem

        print(f"  Source pickle: {source_pickle_rel}")
        print(f"  Objective: {cp_objective}")
        print(f"  Instances: {len(cp_data['instances'])}")

        # 2. 원본 problem pickle 로드
        source_path = project_root / source_pickle_rel
        if not source_path.exists():
            source_name = Path(source_pickle_rel).name
            source_path = project_root / IL_INSTANCE_DIR / source_name
        if not source_path.exists():
            print(f"  원본 problem pickle 찾을 수 없음: {source_pickle_rel}")
            print(f"  시도한 경로: {project_root / source_pickle_rel}, {source_path}")
            continue

        with open(source_path, 'rb') as f:
            problem = pickle.load(f)

        # 3. env_params 구성 (사용자 옵션 적용)
        env_params = {
            'batch_size': 1,
            'pomo_size': 1,
            'objective': cp_objective,
            'state_mode': 'daniel',  # state 수집하지 않으므로 가벼운 모드 사용
            'debug_env': DEBUG_ENV,
            'allow_wait_release': ALLOW_WAIT_RELEASE,
            'allow_wait_mutex': ALLOW_WAIT_MUTEX,
            'allow_wait_team': ALLOW_WAIT_TEAM,
            'allow_wait_pred': ALLOW_WAIT_PRED,
            'use_mutex_attention': USE_MUTEX_ATTENTION,
        }

        # problem pickle의 env_params에서 사용자 제어 키 이외의 값 병합
        _no_overwrite = ('batch_size', 'pomo_size', 'state_mode', 'debug_env',
                         'objective',
                         'allow_wait_release', 'allow_wait_mutex', 'allow_wait_team',
                         'allow_wait_pred',
                         'use_mutex_attention')
        if 'env_params' in problem:
            for k, v in problem['env_params'].items():
                if k not in _no_overwrite:
                    env_params[k] = v

        # 4. 인스턴스별 replay
        file_results = []

        for inst_data in cp_data['instances']:
            idx = inst_data['instance_idx']

            # 범위 필터링
            if INSTANCE_START is not None and idx < INSTANCE_START:
                continue
            if INSTANCE_END is not None and idx >= INSTANCE_END:
                continue

            cp_obj_val = inst_data.get('obj_val')
            print(f"\n   CP_REPLAY Instance {idx}")
            if cp_obj_val is not None:
                print(f"     CP obj: {cp_obj_val:.4f}")

            replay_start_time = time.time()

            try:
                pairs, diagnostics = replay_cp_solution_instance(
                    cp_data, idx, problem, env_params,
                    use_active_schedule=USE_ACTIVE_SCHEDULE,
                    action_selection=ACTION_SELECTION,
                    state_mode='daniel',
                    collect_states=False,
                    collect_diagnostics=True,
                    device=device,
                    debug_env=DEBUG_ENV,
                )

                replay_runtime = time.time() - replay_start_time
                env = diagnostics['env']
                done = diagnostics['done']
                replay_obj = diagnostics['replay_obj']
                fallback_steps = diagnostics['fallback_steps']
                invalid_steps = diagnostics['invalid_steps']
                decision_log = diagnostics['decision_log']

                has_fallback = len(invalid_steps) > 0 or len(fallback_steps) > 0

                if done:
                    diff = abs(replay_obj - cp_obj_val) if cp_obj_val is not None else None
                    match_str = "DIFF" if has_fallback else "MATCH"

                    print(f"     [CP_REPLAY] Replay obj: {replay_obj:.4f}")
                    if diff is not None:
                        print(f"     [CP_REPLAY] Diff: {diff:.4f} ({match_str})")

                    if invalid_steps:
                        print(f"     {len(invalid_steps)}개 step에서 CP decision 불가:")
                        for s, a, t, reason in invalid_steps[:5]:
                            print(f"        step {s}: act={a}, team={t} ({reason})")

                    if fallback_steps:
                        print(f"     {len(fallback_steps)}개 step에서 team fallback:")
                        for s, a, cp_t, actual_t in fallback_steps[:5]:
                            print(f"        step {s}: act={a}, CP team={cp_t} -> actual team={actual_t}")

                    # Gantt chart 생성
                    if SHOW_GANTT_CHART:
                        try:
                            num_act = env.num_activities[0].item()
                            ts = env.time_scale[0, 0].item()
                            schedule = {}
                            act_to_proj = {}
                            for act_id in range(num_act):
                                if env.activity_reserved[0, act_id].item():
                                    s_t = env.activity_start_time[0, act_id].item() * ts
                                    e_t = env.activity_end_time[0, act_id].item() * ts
                                    t = env.activity_assigned_team[0, act_id].item()
                                    schedule[act_id] = (s_t, e_t, t)
                                act_to_proj[act_id] = env.activity_project[0, act_id].item()

                            proj_due = {}
                            for p in range(env.N_P):
                                proj_due[p] = env.project_due_date[0, p].item() * ts

                            step_order = {}
                            for act_id in range(num_act):
                                ss = env.activity_scheduled_step[0, act_id].item()
                                if ss >= 0:
                                    step_order[act_id] = ss

                            create_gantt_chart_from_schedule(
                                schedule=schedule,
                                activity_to_project=act_to_proj,
                                num_teams=env.N_T,
                                instance_name=f"{pickle_name}_instance_{idx}",
                                algorithm="CP_REPLAY",
                                objective_value=replay_obj,
                                project_due_dates=proj_due,
                                activity_step_order=step_order,
                                objective_type=cp_objective,
                                save_dir=gantt_dir,
                                show=True
                            )
                        except Exception as e:
                            print(f"     Gantt chart 생성 실패: {e}")

                    # Replay decision CSV 저장
                    if SAVE_REPLAY_DECISIONS_CSV:
                        try:
                            num_act = env.num_activities[0].item()
                            ts = env.time_scale[0, 0].item()
                            csv_path = decisions_dir / f"{pickle_name}_instance_{idx}.csv"

                            with open(csv_path, 'w', newline='') as rf:
                                rf.write(f"# pickle={pickle_name} instance={idx} "
                                         f"cp_objective={cp_obj_val} replay_objective={replay_obj} "
                                         f"match={match_str}\n")
                                rwriter = csv.writer(rf)
                                rwriter.writerow(['step', 'activity', 'project', 'team',
                                                  'start_time', 'end_time', 'duration', 'note'])
                                for step_n, act_id_r, team_id_r, note_r in decision_log:
                                    if env.activity_reserved[0, act_id_r].item():
                                        s_t = env.activity_start_time[0, act_id_r].item() * ts
                                        e_t = env.activity_end_time[0, act_id_r].item() * ts
                                        dur_r = e_t - s_t
                                    else:
                                        s_t, e_t, dur_r = '', '', ''
                                    proj_id_r = env.activity_project[0, act_id_r].item()
                                    rwriter.writerow([step_n, act_id_r, proj_id_r, team_id_r,
                                                      s_t, e_t, dur_r, note_r])
                            print(f"     Replay decisions saved: {csv_path.name}")
                        except Exception as e:
                            print(f"     Replay CSV 저장 실패: {e}")

                    file_results.append({
                        'instance': idx,
                        'cp_objective': cp_obj_val,
                        'replay_objective': replay_obj,
                        'diff': diff,
                        'match': match_str,
                        'invalid_steps': len(invalid_steps),
                        'fallback_steps': len(fallback_steps),
                        'runtime': replay_runtime,
                    })
                else:
                    print(f"     Replay 미완료 (step_count={diagnostics['step_count']})")
                    file_results.append({
                        'instance': idx,
                        'cp_objective': cp_obj_val,
                        'replay_objective': None,
                        'diff': None,
                        'match': 'INCOMPLETE',
                        'invalid_steps': len(invalid_steps),
                        'fallback_steps': len(fallback_steps),
                        'runtime': replay_runtime,
                    })

            except Exception as e:
                print(f"     CP_REPLAY 오류: {e}")
                import traceback
                traceback.print_exc()
                file_results.append({
                    'instance': idx,
                    'cp_objective': cp_obj_val,
                    'replay_objective': None,
                    'diff': None,
                    'match': 'ERROR',
                    'invalid_steps': 0,
                    'fallback_steps': 0,
                    'runtime': 0,
                })

        # 파일별 결과 요약
        if file_results:
            match_count = sum(1 for r in file_results if r['match'] == 'MATCH')
            diff_count = sum(1 for r in file_results if r['match'] == 'DIFF')
            replay_objs = [r['replay_objective'] for r in file_results if r['replay_objective'] is not None]
            cp_objs = [r['cp_objective'] for r in file_results if r['cp_objective'] is not None]

            print(f"\n  CP_REPLAY 결과 요약 ({solution_path.name}):")
            print(f"    MATCH: {match_count}/{len(file_results)}")
            print(f"    DIFF:  {diff_count}/{len(file_results)}")
            if replay_objs:
                print(f"    평균 replay {cp_objective}: {np.mean(replay_objs):.4f}")
            if cp_objs:
                print(f"    평균 CP {cp_objective}:     {np.mean(cp_objs):.4f}")

            # Excel 저장
            output_path = session_dir / f'CP_Replay_{pickle_name}.xlsx'
            detailed_data = []
            for r in file_results:
                detailed_data.append({
                    'Instance': r['instance'],
                    f'CP {cp_objective.capitalize()}': r['cp_objective'],
                    f'Replay {cp_objective.capitalize()}': r['replay_objective'],
                    'Diff': r['diff'],
                    'Match': r['match'],
                    'Invalid Steps': r['invalid_steps'],
                    'Fallback Steps': r['fallback_steps'],
                    'Runtime': r['runtime'],
                })

            # 실험 정보 시트
            exp_rows = [
                {'Item': 'Algorithm', 'Value': 'CP_REPLAY'},
                {'Item': 'Source File', 'Value': solution_path.name},
                {'Item': 'Problem Pickle', 'Value': pickle_name},
                {'Item': 'Objective', 'Value': cp_objective},
                {'Item': 'Active Schedule', 'Value': str(USE_ACTIVE_SCHEDULE)},
                {'Item': 'Action Selection', 'Value': ACTION_SELECTION},
                {'Item': 'Num Instances', 'Value': len(file_results)},
                {'Item': 'MATCH Count', 'Value': match_count},
                {'Item': 'DIFF Count', 'Value': diff_count},
            ]
            if replay_objs:
                exp_rows.append({'Item': f'Avg Replay {cp_objective}', 'Value': f'{np.mean(replay_objs):.4f}'})
            if cp_objs:
                exp_rows.append({'Item': f'Avg CP {cp_objective}', 'Value': f'{np.mean(cp_objs):.4f}'})

            exp_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])
            detailed_df = pd.DataFrame(detailed_data)
            num_cols = detailed_df.select_dtypes(include=[np.number]).columns
            detailed_df[num_cols] = detailed_df[num_cols].round(4)

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                exp_df.to_excel(writer, sheet_name='Experiment_Info', index=False)
                detailed_df.to_excel(writer, sheet_name='CP_Replay_Results', index=False)

            print(f"  결과가 {output_path}에 저장되었습니다.")

        all_results.extend(file_results)

    # =================================================================
    # 전체 요약
    # =================================================================
    total_time = time.time() - total_start_time

    print(f"\n{'='*60}")
    print("CP Replay 전체 요약")
    print(f"{'='*60}")

    if all_results:
        total_match = sum(1 for r in all_results if r['match'] == 'MATCH')
        total_diff = sum(1 for r in all_results if r['match'] == 'DIFF')
        total_incomplete = sum(1 for r in all_results if r['match'] == 'INCOMPLETE')
        total_error = sum(1 for r in all_results if r['match'] == 'ERROR')
        all_replay_objs = [r['replay_objective'] for r in all_results if r['replay_objective'] is not None]
        all_cp_objs = [r['cp_objective'] for r in all_results if r['cp_objective'] is not None]

        print(f"  총 인스턴스:      {len(all_results)}")
        print(f"  MATCH:            {total_match}")
        print(f"  DIFF:             {total_diff}")
        if total_incomplete > 0:
            print(f"  INCOMPLETE:       {total_incomplete}")
        if total_error > 0:
            print(f"  ERROR:            {total_error}")
        if all_replay_objs:
            print(f"  평균 replay obj:  {np.mean(all_replay_objs):.4f}")
        if all_cp_objs:
            print(f"  평균 CP obj:      {np.mean(all_cp_objs):.4f}")
    else:
        print("  결과 없음")

    print(f"  소요 시간:        {total_time:.1f}s")
    print(f"  결과 폴더:        {session_dir}")
    print("\nDone.")
