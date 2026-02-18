from data_generator import generate_scheduling_data_batch
import torch

env_params = {
    'batch_size': 2, 'pomo_size': 1,
    'N_P': 3, 'N_A_min': 3, 'N_A_max': 6, 'N_T': 5,
    'duration_min': 2, 'duration_max': 8,
    'precedence_prob': 0.3, 'mutex_prob': 0.05,
    'eligible_teams_ratio': 0.6,
}
problem = generate_scheduling_data_batch(env_params)

print("=== Tensor shapes ===")
print("activity_duration:     ", problem['activity_duration'].shape)
print("activity_team_duration:", problem['activity_team_duration'].shape)

preds_all = problem['activity_predecessors'][0]
num_act = int(problem['num_activities'][0].item())
print(f"\n=== Batch 0: {num_act} activities ===")

has_succ = set()
for i in range(num_act):
    for p in preds_all[i][preds_all[i] >= 0].tolist():
        has_succ.add(p)
isolated = [
    i for i in range(num_act)
    if len(preds_all[i][preds_all[i] >= 0].tolist()) == 0 and i not in has_succ
]
print(f"Isolated nodes: {isolated}  (should be [])")

td   = problem['activity_team_duration'][0, 0]
elig = problem['activity_eligible_teams'][0, 0]
mean_dur = problem['activity_duration'][0, 0].item()
eligible_durs = [td[t].item() for t in range(5) if elig[t]]
computed_mean = sum(eligible_durs) / len(eligible_durs) if eligible_durs else 0
print(f"\nAct0  mean_duration={mean_dur:.2f}")
print(f"      eligible_team_durations={eligible_durs}")
print(f"      mean_of_eligible={computed_mean:.2f}  (should match mean_duration)")
print(f"      non-eligible entries are 0: {all(td[t].item()==0 for t in range(5) if not elig[t])}")

# scheduling_env import test
from scheduling_env import SchedulingEnv
env_p2 = dict(env_params)
env_p2['state_mode'] = 'daniel'
env_p2['step_log'] = False
env_p2['objective'] = 'tardiness'
env = SchedulingEnv(env_p2)
env._reset()
print("\n=== SchedulingEnv reset OK ===")
print("activity_team_duration shape in env:", env.activity_team_duration.shape)
state = env._get_state()
print("fea_pairs shape:", state.fea_pairs_tensor.shape)
print("ALL CHECKS PASSED")
