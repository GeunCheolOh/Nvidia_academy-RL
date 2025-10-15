"""
Rollout Buffer for PPO (Multi-Discrete Action Space)
"""
import torch
import numpy as np
from typing import Generator, Tuple


class RolloutBufferMultiDiscrete:
    """
    PPO용 Rollout Buffer (MultiDiscrete 행동 지원)
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_shape: tuple = (3,),  # MultiDiscrete [3, 3, 2]의 shape
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_shape = action_shape
        self.device = device
        
        # 버퍼 초기화
        self.observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *action_shape), dtype=np.int64)  # (buffer_size, 3)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # GAE용
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,  # shape: (3,)
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """경험 추가"""
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """GAE 계산"""
        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values
    
    def get(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """미니배치 생성"""
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)
        
        for start_idx in range(0, self.buffer_size, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                torch.FloatTensor(self.observations[batch_indices]).to(self.device),
                torch.LongTensor(self.actions[batch_indices]).to(self.device),  # shape: (batch, 3)
                torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                torch.FloatTensor(self.returns[batch_indices]).to(self.device),
                torch.FloatTensor(self.values[batch_indices]).to(self.device),
            )
    
    def reset(self):
        """버퍼 초기화"""
        self.pos = 0
        self.full = False

