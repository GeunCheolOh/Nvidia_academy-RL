"""
Self-Play Trainer for Multi-Discrete Action Space
"""
import torch
import numpy as np
from typing import Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from env.pikachu_env import PikachuVolleyballEnvMultiDiscrete
from env.symmetry import mirror_observation
from agents.ppo import PPOAgentMultiDiscrete
from agents.rollout_buffer import RolloutBufferMultiDiscrete


def mirror_action_multidiscrete(action: np.ndarray) -> np.ndarray:
    """
    MultiDiscrete 행동 좌우 반전
    
    Args:
        action: [x_idx, y_idx, power] where x_idx in {0,1,2}
    
    Returns:
        mirrored_action: [x_idx_mirrored, y_idx, power]
            - 0 (left) ↔ 2 (right)
            - 1 (stay) → 1 (stay)
    """
    mirrored = action.copy()
    # x_direction 반전: 0↔2, 1→1
    if action[0] == 0:
        mirrored[0] = 2
    elif action[0] == 2:
        mirrored[0] = 0
    # y_direction, power_hit는 그대로
    return mirrored


class SelfPlayTrainerMultiDiscrete:
    """Multi-Discrete용 Self-Play 트레이너"""
    
    def __init__(
        self,
        env: PikachuVolleyballEnvMultiDiscrete,
        agent: PPOAgentMultiDiscrete,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cuda",
    ):
        self.env = env
        self.agent = agent
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Rollout buffers
        self.buffer_p1 = RolloutBufferMultiDiscrete(n_steps, 15, action_shape=(3,), device=device)
        self.buffer_p2 = RolloutBufferMultiDiscrete(n_steps, 15, action_shape=(3,), device=device)
        
        # 통계
        self.episode_rewards_p1 = []
        self.episode_rewards_p2 = []
        self.episode_lengths = []
        self.episode_scores_p1 = []
        self.episode_scores_p2 = []
    
    def collect_rollouts(self) -> Dict[str, float]:
        """Rollout 수집"""
        (obs_p1, obs_p2), _ = self.env.reset()
        
        episode_reward_p1 = 0.0
        episode_reward_p2 = 0.0
        episode_length = 0
        
        for step in range(self.n_steps):
            # 행동 선택
            action_p1, log_prob_p1, value_p1 = self.agent.select_action(obs_p1)
            action_p2_mirrored, log_prob_p2, value_p2 = self.agent.select_action(obs_p2)
            action_p2 = mirror_action_multidiscrete(action_p2_mirrored)
            
            # 환경 진행
            (next_obs_p1, next_obs_p2), (reward_p1, reward_p2), terminated, truncated, info = \
                self.env.step((action_p1, action_p2))
            
            done = terminated or truncated
            
            # 버퍼에 저장 (action을 그대로 저장 - 3차원 배열)
            self.buffer_p1.add(obs_p1, action_p1, reward_p1, value_p1, log_prob_p1, done)
            self.buffer_p2.add(obs_p2, action_p2_mirrored, reward_p2, value_p2, log_prob_p2, done)
            
            # 업데이트
            obs_p1 = next_obs_p1
            obs_p2 = next_obs_p2
            episode_reward_p1 += reward_p1
            episode_reward_p2 += reward_p2
            episode_length += 1
            self.agent.total_timesteps += 1
            
            if done:
                self.episode_rewards_p1.append(episode_reward_p1)
                self.episode_rewards_p2.append(episode_reward_p2)
                self.episode_lengths.append(episode_length)
                self.episode_scores_p1.append(info['score_p1'])
                self.episode_scores_p2.append(info['score_p2'])
                
                (obs_p1, obs_p2), _ = self.env.reset()
                episode_reward_p1 = 0.0
                episode_reward_p2 = 0.0
                episode_length = 0
        
        # 마지막 value 계산
        with torch.no_grad():
            obs_tensor_p1 = torch.FloatTensor(obs_p1).unsqueeze(0).to(self.device)
            obs_tensor_p2 = torch.FloatTensor(obs_p2).unsqueeze(0).to(self.device)
            last_value_p1 = self.agent.network.get_value(obs_tensor_p1).item()
            last_value_p2 = self.agent.network.get_value(obs_tensor_p2).item()
        
        # GAE 계산
        self.buffer_p1.compute_returns_and_advantages(last_value_p1, self.gamma, self.gae_lambda)
        self.buffer_p2.compute_returns_and_advantages(last_value_p2, self.gamma, self.gae_lambda)
        
        # 통계
        stats = {
            "mean_reward_p1": np.mean(self.episode_rewards_p1[-10:]) if self.episode_rewards_p1 else 0.0,
            "mean_reward_p2": np.mean(self.episode_rewards_p2[-10:]) if self.episode_rewards_p2 else 0.0,
            "mean_length": np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0.0,
            "mean_score_p1": np.mean(self.episode_scores_p1[-10:]) if self.episode_scores_p1 else 0.0,
            "mean_score_p2": np.mean(self.episode_scores_p2[-10:]) if self.episode_scores_p2 else 0.0,
        }
        
        return stats
    
    def train(
        self,
        total_timesteps: int,
        n_epochs: int = 10,
        batch_size: int = 64,
        save_freq: int = 10000,
        eval_freq: int = 10000,
        log_freq: int = 1000,
        save_dir: str = "models/multidiscrete",
        log_dir: str = "logs/multidiscrete",
        resume_from: str = None,
    ):
        """학습 실행"""
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        if resume_from and os.path.exists(resume_from):
            print(f"\n체크포인트에서 재개: {resume_from}")
            self.agent.load(resume_from)
        
        start_timestep = self.agent.total_timesteps
        pbar = tqdm(total=total_timesteps - start_timestep, desc="Training")
        
        # Best model 추적
        best_score = 0.0
        
        while self.agent.total_timesteps < total_timesteps:
            # Rollout 수집
            rollout_stats = self.collect_rollouts()
            
            # 학습
            update_stats_p1 = self.agent.update(self.buffer_p1, n_epochs, batch_size)
            update_stats_p2 = self.agent.update(self.buffer_p2, n_epochs, batch_size)
            
            # 버퍼 리셋
            self.buffer_p1.reset()
            self.buffer_p2.reset()
            
            # 통계 평균
            update_stats = {
                key: (update_stats_p1[key] + update_stats_p2[key]) / 2
                for key in update_stats_p1.keys()
            }
            
            # 로깅
            if self.agent.total_timesteps % log_freq < self.n_steps:
                for key, value in rollout_stats.items():
                    writer.add_scalar(f"Rollout/{key}", value, self.agent.total_timesteps)
                
                for key, value in update_stats.items():
                    writer.add_scalar(f"Train/{key}", value, self.agent.total_timesteps)
                
                pbar.set_postfix({
                    "reward_p1": f"{rollout_stats['mean_reward_p1']:.2f}",
                    "score": f"{rollout_stats['mean_score_p1']:.1f}-{rollout_stats['mean_score_p2']:.1f}",
                    "loss": f"{update_stats['total_loss']:.3f}",
                })
            
            # 저장
            if self.agent.total_timesteps % save_freq < self.n_steps:
                checkpoint_path = f"{save_dir}/checkpoint_{self.agent.total_timesteps}.pth"
                self.agent.save(checkpoint_path)
                print(f"\n체크포인트 저장: {checkpoint_path}")
            
            # 평가
            if self.agent.total_timesteps % eval_freq < self.n_steps:
                eval_stats = self.evaluate(num_episodes=10)
                for key, value in eval_stats.items():
                    writer.add_scalar(f"Eval/{key}", value, self.agent.total_timesteps)
                
                print(f"\n평가 결과 (timestep {self.agent.total_timesteps}):")
                print(f"  평균 점수: P1={eval_stats['mean_score_p1']:.1f}, P2={eval_stats['mean_score_p2']:.1f}")
                print(f"  승률 P1: {eval_stats['win_rate_p1']:.1%}")
                
                # Best model 저장
                current_score = eval_stats['mean_score_p1']
                if current_score > best_score:
                    best_score = current_score
                    best_path = f"{save_dir}/best_model.pth"
                    self.agent.save(best_path)
                    print(f"  새로운 Best Model 저장! Score: {current_score:.2f} (이전: {best_score:.2f})")
            
            pbar.update(self.n_steps)
        
        pbar.close()
        writer.close()
        
        # 최종 모델 저장
        final_path = f"{save_dir}/final_model.pth"
        self.agent.save(final_path)
        print(f"\n최종 모델 저장: {final_path}")
        print(f"Best Score 달성: {best_score:.2f}")
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """에이전트 평가"""
        scores_p1 = []
        scores_p2 = []
        wins_p1 = 0
        
        for _ in range(num_episodes):
            (obs_p1, obs_p2), _ = self.env.reset()
            done = False
            
            while not done:
                action_p1, _, _ = self.agent.select_action(obs_p1, deterministic=True)
                action_p2_mirrored, _, _ = self.agent.select_action(obs_p2, deterministic=True)
                action_p2 = mirror_action_multidiscrete(action_p2_mirrored)
                
                (obs_p1, obs_p2), _, terminated, truncated, info = \
                    self.env.step((action_p1, action_p2))
                
                done = terminated or truncated
            
            scores_p1.append(info['score_p1'])
            scores_p2.append(info['score_p2'])
            if info['score_p1'] > info['score_p2']:
                wins_p1 += 1
        
        return {
            "mean_score_p1": np.mean(scores_p1),
            "mean_score_p2": np.mean(scores_p2),
            "win_rate_p1": wins_p1 / num_episodes,
        }

