"""
Actor-Critic Network for PPO (Multi-Discrete Action Space)

Actor: 3개의 독립적인 헤드
  - x_direction: Categorical(3)
  - y_direction: Categorical(3)
  - power_hit: Categorical(2)
Critic: 상태 가치 추정
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple


class ActorCriticNetworkMultiDiscrete(nn.Module):
    """
    Multi-Discrete Action Space용 Actor-Critic 네트워크
    
    Action = (x_direction, y_direction, power_hit)
    """
    
    def __init__(
        self,
        observation_dim: int = 15,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        
        # 공유 Feature Extractor
        layers = []
        prev_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor Heads (각 차원별로 독립적)
        self.x_direction_head = nn.Linear(prev_dim, 3)  # x_direction: 3가지
        self.y_direction_head = nn.Linear(prev_dim, 3)  # y_direction: 3가지
        self.power_hit_head = nn.Linear(prev_dim, 2)  # power_hit: 2가지
        
        # Critic Head
        self.critic = nn.Linear(prev_dim, 1)
        
        # 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            x_logits: (batch, 3) x_direction 로짓
            y_logits: (batch, 3) y_direction 로짓
            power_logits: (batch, 2) power_hit 로짓
            value: (batch, 1) 상태 가치
        """
        features = self.shared_net(x)
        
        x_logits = self.x_direction_head(features)
        y_logits = self.y_direction_head(features)
        power_logits = self.power_hit_head(features)
        value = self.critic(features)
        
        return x_logits, y_logits, power_logits, value
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        행동 샘플링 및 가치 추정
        
        Args:
            x: (batch, observation_dim) 관찰
            action: (batch, 3) 행동 [x, y, power] (None이면 샘플링)
        
        Returns:
            action: (batch, 3) 선택된 행동
            log_prob: (batch,) 행동의 로그 확률
            entropy: (batch,) 정책 엔트로피
            value: (batch, 1) 상태 가치
        """
        x_logits, y_logits, power_logits, value = self.forward(x)
        
        # 각 차원에 대한 분포
        dist_x = Categorical(logits=x_logits)
        dist_y = Categorical(logits=y_logits)
        dist_power = Categorical(logits=power_logits)
        
        if action is None:
            # 각 차원에서 독립적으로 샘플링
            action_x = dist_x.sample()
            action_y = dist_y.sample()
            action_power = dist_power.sample()
            action = torch.stack([action_x, action_y, action_power], dim=-1)
        else:
            action_x = action[:, 0]
            action_y = action[:, 1]
            action_power = action[:, 2]
        
        # 로그 확률 (각 차원의 로그 확률 합)
        log_prob_x = dist_x.log_prob(action_x)
        log_prob_y = dist_y.log_prob(action_y)
        log_prob_power = dist_power.log_prob(action_power)
        log_prob = log_prob_x + log_prob_y + log_prob_power
        
        # 엔트로피 (각 차원의 엔트로피 합)
        entropy = dist_x.entropy() + dist_y.entropy() + dist_power.entropy()
        
        return action, log_prob, entropy, value
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """상태 가치만 추정"""
        features = self.shared_net(x)
        value = self.critic(features)
        return value

