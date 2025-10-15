"""
PPO Agent for Multi-Discrete Action Space
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

from .network import ActorCriticNetworkMultiDiscrete
from .rollout_buffer import RolloutBufferMultiDiscrete


class PPOAgentMultiDiscrete:
    """Multi-Discrete Action Space용 PPO 에이전트"""
    
    def __init__(
        self,
        observation_dim: int = 15,
        device: str = "cuda",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_value: bool = True,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_value = clip_value
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        
        # 네트워크
        self.network = ActorCriticNetworkMultiDiscrete(
            observation_dim=observation_dim,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )
        
        # 학습 통계
        self.total_timesteps = 0
        self.num_updates = 0
    
    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple:
        """
        행동 선택
        
        Returns:
            action: (3,) numpy array [x, y, power]
            log_prob: float
            value: float
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            if deterministic:
                # Greedy 선택 (각 차원에서 최대 확률)
                x_logits, y_logits, power_logits, value = self.network(obs_tensor)
                action_x = x_logits.argmax(dim=-1)
                action_y = y_logits.argmax(dim=-1)
                action_power = power_logits.argmax(dim=-1)
                action = torch.stack([action_x, action_y, action_power], dim=-1)
                log_prob = torch.zeros(1)
            else:
                # 확률적 선택
                action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
            
            return action.cpu().numpy()[0], log_prob.item(), value.item()
    
    def update(
        self,
        rollout_buffer: RolloutBufferMultiDiscrete,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """PPO 업데이트"""
        # 통계
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fractions = []
        
        for epoch in range(n_epochs):
            for batch in rollout_buffer.get(batch_size):
                (
                    obs_batch,
                    action_batch,
                    old_log_prob_batch,
                    advantage_batch,
                    return_batch,
                    old_value_batch,
                ) = batch
                
                # Advantage 정규화
                if self.normalize_advantages:
                    advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
                
                # 현재 정책으로 평가
                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    obs_batch, action_batch.long()
                )
                
                # Policy loss
                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.clip_value:
                    value_pred_clipped = old_value_batch + torch.clamp(
                        new_value.squeeze() - old_value_batch,
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_loss1 = (new_value.squeeze() - return_batch).pow(2)
                    value_loss2 = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * (new_value.squeeze() - return_batch).pow(2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 역전파
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 통계 수집
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
                
                # KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    approx_kls.append(approx_kl.item())
                    
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    clip_fractions.append(clip_fraction.item())
        
        self.num_updates += 1
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions),
        }
    
    def save(self, path: str):
        """모델 저장"""
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_timesteps": self.total_timesteps,
            "num_updates": self.num_updates,
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint["total_timesteps"]
        self.num_updates = checkpoint["num_updates"]
        
        print(f"체크포인트 로드 완료:")
        print(f"  Total timesteps: {self.total_timesteps:,}")
        print(f"  Num updates: {self.num_updates:,}")

