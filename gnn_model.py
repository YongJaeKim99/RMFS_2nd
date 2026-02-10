"""
RCMPSP (Resource-Constrained Multi-Project Scheduling Problem) GNN 모델
GAT (Graph Attention Network) 기반 정책 네트워크
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch


class SchedulingModel(nn.Module):
    """
    RCMPSP 환경을 위한 GNN 기반 모델
    
    노드 타입:
        - Activity: 스케줄링할 작업
        - Team: 리소스
        - Project: 프로젝트 (Activity의 그룹)
    
    엣지 타입:
        1. Activity → Activity (Precedence): 선행 관계
        2. Activity ↔ Activity (Mutex): 동시 불가
        3. Activity → Team (Eligible): 작업 가능한 팀
        4. Activity → Project (Belongs-to): 소속 프로젝트
    
    Action:
        - (Activity, Team) 페어 선택
    """
    
    def __init__(self, model_params, debug_model=False):
        super(SchedulingModel, self).__init__()
        self.model_params = model_params
        self.debug_model = debug_model
        
        # GNN 모델
        self.gnn_model = GNNModel(self.model_params)
    
    def set_action_space(self, action_to_pair, max_action_space):
        """
        환경에서 생성한 action space 정보를 모델에 전달
        
        Args:
            action_to_pair: (batch_size, max_action_space, 2) - action index → (activity, team) 매핑
            max_action_space: int - 배치 내 최대 action space 크기
        """
        self.gnn_model.set_action_space(action_to_pair, max_action_space)

    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: PyG Batch 객체
        
        Returns:
            action_logits: (total_actions,) - masked logit
        """
        # GNNModel 사용
        action_logits = self.gnn_model(state.x, state.edge_index, state)
        
        # Reshape and apply mask
        action_logits = action_logits.reshape(-1)
        
        # 마스크 적용: invalid action에 log(0) = -inf 추가
        mask_float = state.mask.float()
        action_logits = action_logits + torch.log(mask_float + 1e-10)
        
        return action_logits
    
    def get_prob(self, s, batch):
        """
        확률 분포 반환
        
        Args:
            s: state
            batch: 배치 크기
            
        Returns:
            action_prob: 액션에 대한 확률 분포
        """
        device = next(self.parameters()).device
        s = s.to(device)
        action_logits = self.forward(s)
        action_logits = action_logits.reshape(batch, -1)
        action_prob = F.softmax(action_logits, -1)
        
        return action_prob

    def get_action(self, state):
        """
        액션 샘플링
        
        Args:
            state: 환경 정보를 포함하는 state 객체 리스트
            
        Returns:
            action: 선택된 (Activity, Team) 조합 액션
            log_prob: 액션의 로그 확률
            entropy: 정책의 entropy
        """
        x = DataLoader(state, batch_size=len(state))
        for s in x:
            break
        
        device = next(self.parameters()).device
        s = s.to(device)
        
        # Forward pass
        action_logits = self.forward(s)
        action_logits = action_logits.reshape(len(state), -1)
        
        # 액션 샘플링 (각 배치별로)
        action_list = []
        log_prob_list = []
        entropy_list = []
        
        for b in range(len(state)):
            # 액션 선택 - 마스크 적용
            logit_b = action_logits[b]
            mask_b = state[b].mask
            
            # 마스크를 logit 크기에 맞춰서 조정
            if len(mask_b) > len(logit_b):
                mask_b = mask_b[:len(logit_b)]
            
            num_available = mask_b.sum().item()
            
            # 디버깅
            if self.debug_model:
                available_indices = torch.where(mask_b)[0]
                print(f"\n[DEBUG] Batch {b} - Available actions: {num_available}")
                print(f"  Logit size: {len(logit_b)}, Mask size: {len(mask_b)}")
                if num_available > 0 and num_available <= 20:
                    print(f"  Available action indices: {available_indices.tolist()}")
            
            # 모든 action이 infeasible인 경우
            if num_available == 0:
                action_b = torch.tensor(0, device=device)
                log_prob_b = torch.tensor(0.0, device=device)
                entropy_b = torch.tensor(0.0, device=device)
                
                action_list.append(action_b)
                log_prob_list.append(log_prob_b)
                entropy_list.append(entropy_b)
                continue
            
            # 마스크 적용
            mask_b = mask_b.to(logit_b.device)
            if logit_b.device.type == 'cuda':
                logit_b = torch.where(mask_b, logit_b, torch.tensor(-1e9, device=logit_b.device))
            else:
                logit_b = logit_b.clone()
                logit_b[~mask_b] = -1e9
            
            prob_b = F.softmax(logit_b, dim=0)
            policy_b = Categorical(prob_b)
            action_b = policy_b.sample()
            log_prob_b = policy_b.log_prob(action_b)
            entropy_b = policy_b.entropy()
            
            # 디버깅
            if self.debug_model and b < 3:
                print(f"[DEBUG] Batch {b}: Action={action_b.item()}, Prob={prob_b[action_b].item():.4f}")
            
            action_list.append(action_b)
            log_prob_list.append(log_prob_b)
            entropy_list.append(entropy_b)
        
        # 텐서로 변환
        action = torch.stack(action_list)
        log_prob = torch.stack(log_prob_list)
        entropy = torch.stack(entropy_list)
        
        return action, log_prob, entropy

    def get_max_action(self, state):
        """
        최대 확률의 액션 선택 (greedy)
        
        Args:
            state: 환경 정보를 포함하는 state 객체 리스트
            
        Returns:
            action: 선택된 액션 (greedy)
        """
        x = DataLoader(state, batch_size=len(state))
        for s in x:
            break
        
        device = next(self.parameters()).device
        s = s.to(device)
        action_logits = self.forward(s)
        action_logits = action_logits.reshape(len(state), -1)
        
        action_list = []
        
        for b in range(len(state)):
            logits_b = action_logits[b]
            mask_b = state[b].mask
            
            # 마스크를 logit 크기에 맞춰서 조정
            if len(mask_b) > len(logits_b):
                mask_b = mask_b[:len(logits_b)]
            
            num_available = mask_b.sum().item()
            
            # 모든 action이 infeasible인 경우
            if num_available == 0:
                action_b = torch.tensor(0, device=device)
                action_list.append(action_b)
                continue
            
            # forward()에서 이미 마스크 적용됨
            action_b = torch.argmax(logits_b)
            action_list.append(action_b)
        
        action = torch.stack(action_list)
        
        return action


class GNNModel(nn.Module):
    """
    단순화된 GNN 모델 (단일 엣지 타입)
    모든 엣지를 하나로 취급
    """
    def __init__(self, model_params):
        super(GNNModel, self).__init__()
        self.model_params = model_params
        self.input_dim = self.model_params["input_dim"]
        self.embedding_dim = self.model_params["embedding_dim"]
        self.head = self.model_params["num_head"]
        self.encoder_layer_num = self.model_params["num_encoder_layer"]
        
        # GAT 버전 확인 (기본값: v2)
        self.gat_version = self.model_params.get("gat_version", "v2")
        GATLayer = GATConv if self.gat_version == "v1" else GATv2Conv
        
        D = self.embedding_dim
        H = self.head
        L = self.encoder_layer_num

        # 초기 임베딩 레이어
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim), 
            nn.ReLU()
        )
        
        # GAT 레이어들
        self.gat_layers = nn.ModuleList([
            GATLayer(D, D//H, heads=H, dropout=0.20) 
            for _ in range(L)
        ])

        # Residual connection용 concat 레이어
        self.concat_layers = nn.Linear(2 * D, D)
        
        # 액션 디코딩 레이어 (Activity, Team) - 2개 노드
        self.action_layers = nn.Sequential(
            nn.Linear(2 * D, D), 
            nn.ReLU(),
            nn.Linear(D, 1)
        )
        
        self.relu = nn.ReLU()
        
        # Action space 정보 저장 (환경에서 초기화 시 설정됨)
        self.action_to_pair_cache = None  # (batch_size, max_action_space, 2)
        self.max_action_space = None
    
    def set_action_space(self, action_to_pair, max_action_space):
        """
        환경에서 생성한 action space 정보를 모델에 저장
        
        Args:
            action_to_pair: (batch_size, max_action_space, 2) - action index → (activity, team) 매핑
            max_action_space: int - 배치 내 최대 action space 크기
        """
        self.action_to_pair_cache = action_to_pair
        self.max_action_space = max_action_space

    def forward(self, n_f, edge_index, state):
        """
        GNN forward pass
        
        Args:
            n_f: 노드 피처 (total_nodes, input_dim)
            edge_index: 엣지 인덱스 (2, num_edges)
            state: 전체 state 객체 (Batch 또는 list)
        
        Returns:
            action_logits: (total_actions,) - action logit (eligible한 조합만)
        """
        # (1) Encoder: 노드 임베딩 생성
        emd = self.embedding(n_f)
        
        for layer in range(self.encoder_layer_num):
            out = self.gat_layers[layer](emd, edge_index)
            # Residual connection
            emd = self.relu(
                self.concat_layers(torch.cat([out, emd], dim=1))
            )
        
        # (2) Decoder: 액션 logit 계산
        # Action: (Activity, Team) 페어 - eligible한 조합만 처리
        
        # 환경 파라미터
        N_T = self.model_params.get('N_T', 4)
        N_P = self.model_params.get('N_P', 5)
        
        # 각 배치별로 action decoder 생성
        action_logits_list = []
        
        # PyG Batch 객체인 경우 batch 속성 사용
        if hasattr(state, 'batch'):
            batch_indices = state.batch
            num_graphs = batch_indices.max().item() + 1
            
            # Batch 객체에서 num_activities 리스트 추출
            if hasattr(state, 'num_activities'):
                if isinstance(state.num_activities, torch.Tensor):
                    if state.num_activities.dim() == 0:
                        # 스칼라인 경우 (배치 크기 1)
                        num_activities_list = [state.num_activities.item()]
                    else:
                        # 벡터인 경우 (배치 크기 > 1)
                        num_activities_list = state.num_activities.tolist()
                else:
                    num_activities_list = [state.num_activities] * num_graphs
            else:
                num_activities_list = [0] * num_graphs
            
            # action_to_pair 정보 추출 (캐시 우선, 없으면 state에서)
            action_to_pair_list = None
            if self.action_to_pair_cache is not None:
                action_to_pair_list = self.action_to_pair_cache
            elif hasattr(state, 'action_to_pair'):
                action_to_pair_list = state.action_to_pair
                # 첫 forward 시 캐시에 저장
                self.action_to_pair_cache = action_to_pair_list
                if hasattr(state, 'mask') and isinstance(state.mask, torch.Tensor):
                    self.max_action_space = state.mask.shape[0] if state.mask.dim() == 1 else state.mask.shape[1]
            
            # 각 그래프별로 처리
            for b_idx in range(num_graphs):
                # 이 배치에 속하는 노드들
                node_mask = (batch_indices == b_idx)
                
                # num_activities 가져오기
                num_act = num_activities_list[b_idx]
                
                # Activity와 Team 노드의 임베딩 추출
                graph_emd = emd[node_mask]  # (graph_nodes, D)
                
                act_embeddings = graph_emd[:num_act, :]  # (num_act, D)
                team_embeddings = graph_emd[num_act:num_act+N_T, :]  # (N_T, D)
                
                # Eligible한 (Activity, Team) 페어만 처리
                if action_to_pair_list is not None:
                    action_to_pair = action_to_pair_list[b_idx]  # (max_action_space, 2)
                    
                    # 유효한 action만 선택 (패딩 제외)
                    valid_mask = action_to_pair[:, 0] >= 0
                    valid_actions = action_to_pair[valid_mask]  # (num_valid_actions, 2)
                    
                    # 각 eligible action에 대한 임베딩 생성
                    action_embeddings = []
                    for act_id, team_id in valid_actions:
                        act_emb = act_embeddings[act_id]  # (D,)
                        team_emb = team_embeddings[team_id]  # (D,)
                        action_emb = torch.cat([act_emb, team_emb], dim=0)  # (2*D,)
                        action_embeddings.append(action_emb)
                    
                    if len(action_embeddings) > 0:
                        action_decode_emb = torch.stack(action_embeddings)  # (num_valid_actions, 2*D)
                        logits_b = self.action_layers(action_decode_emb).squeeze(-1)  # (num_valid_actions,)
                        
                        # 패딩된 action에 대해서는 매우 작은 값으로 채움
                        max_action_space = action_to_pair.shape[0]
                        full_logits = torch.full((max_action_space,), -1e9, device=logits_b.device, dtype=logits_b.dtype)
                        full_logits[valid_mask] = logits_b
                        action_logits_list.append(full_logits)
                    else:
                        # 유효한 action이 없는 경우
                        max_action_space = action_to_pair.shape[0]
                        action_logits_list.append(torch.full((max_action_space,), -1e9, device=emd.device))
                else:
                    # action_to_pair 정보가 없으면 기존 방식 사용 (모든 조합)
                    act_emb_expanded = act_embeddings.unsqueeze(1).expand(-1, N_T, -1)  # (num_act, N_T, D)
                    team_emb_expanded = team_embeddings.unsqueeze(0).expand(num_act, -1, -1)  # (num_act, N_T, D)
                    
                    action_decode_emb = torch.cat([act_emb_expanded, team_emb_expanded], dim=2)
                    action_decode_emb = action_decode_emb.reshape(-1, 2 * self.embedding_dim)  # (num_act*N_T, 2*D)
                    
                    logits_b = self.action_layers(action_decode_emb).squeeze(-1)  # (num_act*N_T,)
                    action_logits_list.append(logits_b)
        
        else:
            # List인 경우 (get_action에서 DataLoader를 거친 경우)
            for b_idx in range(len(state)):
                num_act = state[b_idx].num_activities
                
                # 각 그래프의 노드 범위 계산
                graph_nodes = num_act + N_T + N_P
                node_start = b_idx * graph_nodes
                node_end = node_start + graph_nodes
                
                # Activity와 Team 노드의 임베딩 추출
                act_embeddings = emd[node_start:node_start+num_act, :]  # (num_act, D)
                team_embeddings = emd[node_start+num_act:node_start+num_act+N_T, :]  # (N_T, D)
                
                # Eligible한 (Activity, Team) 페어만 처리
                if hasattr(state[b_idx], 'action_to_pair'):
                    action_to_pair = state[b_idx].action_to_pair  # (max_action_space, 2)
                    
                    # 유효한 action만 선택 (패딩 제외)
                    valid_mask = action_to_pair[:, 0] >= 0
                    valid_actions = action_to_pair[valid_mask]  # (num_valid_actions, 2)
                    
                    # 각 eligible action에 대한 임베딩 생성
                    action_embeddings = []
                    for act_id, team_id in valid_actions:
                        act_emb = act_embeddings[act_id]  # (D,)
                        team_emb = team_embeddings[team_id]  # (D,)
                        action_emb = torch.cat([act_emb, team_emb], dim=0)  # (2*D,)
                        action_embeddings.append(action_emb)
                    
                    if len(action_embeddings) > 0:
                        action_decode_emb = torch.stack(action_embeddings)  # (num_valid_actions, 2*D)
                        logits_b = self.action_layers(action_decode_emb).squeeze(-1)  # (num_valid_actions,)
                        
                        # 패딩된 action에 대해서는 매우 작은 값으로 채움
                        max_action_space = action_to_pair.shape[0]
                        full_logits = torch.full((max_action_space,), -1e9, device=logits_b.device, dtype=logits_b.dtype)
                        full_logits[valid_mask] = logits_b
                        action_logits_list.append(full_logits)
                    else:
                        # 유효한 action이 없는 경우
                        max_action_space = action_to_pair.shape[0]
                        action_logits_list.append(torch.full((max_action_space,), -1e9, device=emd.device))
                else:
                    # action_to_pair 정보가 없으면 기존 방식 사용 (모든 조합)
                    act_emb_expanded = act_embeddings.unsqueeze(1).expand(-1, N_T, -1)  # (num_act, N_T, D)
                    team_emb_expanded = team_embeddings.unsqueeze(0).expand(num_act, -1, -1)  # (num_act, N_T, D)
                    
                    action_decode_emb = torch.cat([act_emb_expanded, team_emb_expanded], dim=2)
                    action_decode_emb = action_decode_emb.reshape(-1, 2 * self.embedding_dim)  # (num_act*N_T, 2*D)
                    
                    logits_b = self.action_layers(action_decode_emb).squeeze(-1)  # (num_act*N_T,)
                    action_logits_list.append(logits_b)
        
        # 모든 배치의 logits를 concat
        action_logits = torch.cat(action_logits_list, dim=0)
        
        return action_logits


# ========================================
# 테스트 코드
# ========================================
if __name__ == "__main__":
    print("="*60)
    print("SchedulingModel 테스트")
    print("="*60)
    
    # 모델 파라미터
    model_params = {
        'embedding_dim': 128,
        'num_head': 8,
        'num_encoder_layer': 3,
        'input_dim': 10,
        'gat_version': 'v2',
        'N_T': 3,
        'N_P': 3,
    }
    
    # 모델 생성
    model = SchedulingModel(model_params, debug_model=True)
    print(f"\n✅ 모델 생성 완료")
    print(f"   Embedding Dim: {model_params['embedding_dim']}")
    print(f"   Num Heads: {model_params['num_head']}")
    print(f"   Num Layers: {model_params['num_encoder_layer']}")
    
    # 더미 state 생성
    num_activities = 10
    N_T = 3
    N_P = 3
    num_nodes = num_activities + N_T + N_P
    
    # 노드 feature (10차원)
    x = torch.randn(num_nodes, 10)
    
    # 엣지 (랜덤)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    
    # Action mask (일부만 True)
    mask = torch.zeros(num_activities * N_T, dtype=torch.bool)
    mask[:10] = True  # 처음 10개만 가능
    
    # PyG Data 객체
    data = Data(
        x=x,
        edge_index=edge_index,
        mask=mask,
        num_activities=num_activities,
        N_T=N_T,
        N_P=N_P
    )
    
    state = [data]
    
    # Forward pass 테스트
    print(f"\n🔄 Forward pass 테스트...")
    action, log_prob, entropy = model.get_action(state)
    print(f"   Action: {action.item()}")
    print(f"   Log Prob: {log_prob.item():.4f}")
    print(f"   Entropy: {entropy.item():.4f}")
    
    # Greedy action 테스트
    print(f"\n🎯 Greedy action 테스트...")
    greedy_action = model.get_max_action(state)
    print(f"   Greedy Action: {greedy_action.item()}")
    
    print("\n✅ SchedulingModel 테스트 완료!")
