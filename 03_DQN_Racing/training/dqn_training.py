#!/usr/bin/env python3
"""
CarRacing 환경을 위한 DQN 학습 스크립트

이 스크립트는 CarRacing-v2 환경을 위한 DQN(Deep Q-Networks) 학습 파이프라인을 구현합니다.
주요 DQN 구성 요소를 모두 포함합니다:
- CNN 기반 Q-Network
- Experience Replay Buffer  
- Target Network
- Epsilon-Greedy Strategy
- 모니터링을 포함한 학습 루프

사용법:
    python dqn_training.py [--episodes 500] [--render] [--load model.pth]

작성자: DQN Racing Tutorial
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import cv2
import random
import argparse
import os
import time
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# 하이퍼파라미터 설정
# ============================================================================

HYPERPARAMETERS = {
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'batch_size': 32,
    'buffer_size': 10000,
    'target_update': 1000,
    'num_episodes': 500,
    'max_steps_per_episode': 1000,
    'frame_stack': 4,
    'image_size': (84, 84),
    'seed': 42,
    'save_interval': 50,
    'log_interval': 10
}


# ============================================================================
# 환경 전처리
# ============================================================================

class CarRacingWrapper:
    """전처리가 포함된 CarRacing 환경 래퍼"""
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        CarRacing 환경 래퍼 초기화
        
        Args:
            render_mode: 렌더링 모드 ('human', 'rgb_array', 또는 None)
        """
        self.env = gym.make('CarRacing-v3', render_mode=render_mode)
        self.frame_stack = HYPERPARAMETERS['frame_stack']
        self.image_size = HYPERPARAMETERS['image_size']
        
        # 프레임 스태킹을 위한 버퍼
        self.frames = deque(maxlen=self.frame_stack)
        
    def reset(self) -> np.ndarray:
        """환경 리셋 및 초기 스택 프레임 반환"""
        obs, info = self.env.reset()
        
        # 초기 프레임 전처리
        processed_frame = self._preprocess_frame(obs)
        
        # 첫 프레임을 반복하여 프레임 스택 초기화
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
            
        return self._get_stacked_frames()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        행동을 수행하고 전처리된 관측값 반환
        
        Args:
            action: 이산 행동 인덱스
            
        Returns:
            (관측값, 보상, 종료여부, 잘림여부, 정보) 튜플
        """
        # 이산 행동을 연속 행동으로 변환
        continuous_action = self._discrete_to_continuous(action)
        
        # 환경에서 스텝 수행
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        # 프레임 전처리 및 스택
        processed_frame = self._preprocess_frame(obs)
        self.frames.append(processed_frame)
        stacked_frames = self._get_stacked_frames()
        
        return stacked_frames, reward, terminated, truncated, info
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 전처리: 크기 조정, 그레이스케일, 정규화
        
        Args:
            frame: 환경에서 받은 원본 프레임
            
        Returns:
            전처리된 프레임
        """
        # 그레이스케일로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 타겟 크기로 리사이즈
        resized_frame = cv2.resize(gray_frame, self.image_size)
        
        # [0, 1] 범위로 정규화
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        return normalized_frame
        
    def _get_stacked_frames(self) -> np.ndarray:
        """스택된 프레임을 numpy 배열로 반환"""
        return np.array(list(self.frames))
        
    def _discrete_to_continuous(self, action: int) -> np.ndarray:
        """
        이산 행동을 연속 행동 공간으로 변환
        
        Args:
            action: 이산 행동 (0=왼쪽, 1=직진, 2=오른쪽, 3=브레이크)
            
        Returns:
            연속 행동 [조향, 가스, 브레이크]
        """
        if action == 0:     # 왼쪽 회전
            return np.array([-0.5, 0.3, 0.0])
        elif action == 1:   # 직진
            return np.array([0.0, 0.5, 0.0])
        elif action == 2:   # 오른쪽 회전
            return np.array([0.5, 0.3, 0.0])
        elif action == 3:   # 브레이크
            return np.array([0.0, 0.0, 0.8])
        else:
            return np.array([0.0, 0.0, 0.0])
            
    def close(self):
        """환경 종료"""
        self.env.close()


# ============================================================================
# DQN 네트워크 구조
# ============================================================================

class DQN(nn.Module):
    """CarRacing을 위한 CNN 기반 Deep Q-Network"""
    
    def __init__(self, action_dim: int = 4, input_channels: int = 4):
        """
        DQN 네트워크 초기화
        
        Args:
            action_dim: 이산 행동의 개수
            input_channels: 입력 채널 수 (프레임 스택)
        """
        super(DQN, self).__init__()
        
        # 합성곱 레이어
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Conv 출력 크기 계산
        self._conv_output_size = self._get_conv_output_size((input_channels, 84, 84))
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_dim)
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _get_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Conv 레이어 통과 후 출력 크기 계산"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self._forward_conv(dummy_input)
            return dummy_output.numel()
            
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Conv 레이어만 통과하는 forward pass"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)
        
    def _initialize_weights(self):
        """네트워크 가중치 초기화"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        네트워크 forward pass
        
        Args:
            x: 입력 텐서 (batch_size, channels, height, width)
            
        Returns:
            각 행동에 대한 Q-값
        """
        # TODO: DQN 네트워크의 forward pass를 구현하세요
        # 힌트 1: self._forward_conv(x)로 Conv 레이어를 통과시킵니다
        # 힌트 2: F.relu를 사용하여 fc1 레이어를 통과시킵니다  
        # 힌트 3: fc2 레이어를 통과시켜 최종 Q-값들을 출력합니다
        # 힌트 4: Q-값은 각 행동의 예상 가치를 나타냅니다
        #YOUR CODE HERE
        raise NotImplementedError("DQN forward pass를 구현하세요")


# ============================================================================
# Experience Replay Buffer
# ============================================================================

class ReplayBuffer:
    """전이(transition) 저장을 위한 경험 재생 버퍼"""
    
    def __init__(self, capacity: int):
        """
        Replay buffer 초기화
        
        Args:
            capacity: 저장할 전이의 최대 개수
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, reward, next_state, done):
        """버퍼에 전이 추가"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """전이 배치 샘플링"""
        # TODO: Replay Buffer에서 배치를 샘플링하세요
        # 힌트 1: random.sample을 사용하여 buffer에서 batch_size만큼 샘플링합니다
        # 힌트 2: zip(*batch)로 states, actions, rewards, next_states, dones를 분리합니다
        # 힌트 3: 각각을 적절한 torch 텐서로 변환합니다 (FloatTensor, LongTensor, BoolTensor)
        # 힌트 4: Experience Replay는 샘플 간 상관관계를 줄여 학습을 안정화합니다
        #YOUR CODE HERE
        raise NotImplementedError("Replay Buffer sampling을 구현하세요")
        
    def __len__(self):
        """현재 버퍼 크기 반환"""
        return len(self.buffer)


# ============================================================================
# DQN 에이전트
# ============================================================================

class DQNAgent:
    """모든 학습 구성요소를 포함한 DQN 에이전트"""
    
    def __init__(self, device: torch.device):
        """
        DQN 에이전트 초기화
        
        Args:
            device: 연산을 수행할 장치
        """
        self.device = device
        self.action_dim = 4  # 왼쪽, 직진, 오른쪽, 브레이크
        
        # 네트워크
        self.main_network = DQN(self.action_dim).to(device)
        self.target_network = DQN(self.action_dim).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # 옵티마이저
        self.optimizer = optim.Adam(
            self.main_network.parameters(), 
            lr=HYPERPARAMETERS['learning_rate']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(HYPERPARAMETERS['buffer_size'])
        
        # 탐험 전략
        self.epsilon = HYPERPARAMETERS['epsilon_start']
        self.epsilon_decay = HYPERPARAMETERS['epsilon_decay']
        self.epsilon_min = HYPERPARAMETERS['epsilon_end']
        
        # 학습 카운터
        self.step_count = 0
        self.episode_count = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy 정책으로 행동 선택
        
        Args:
            state: 현재 상태
            training: 학습 모드 여부
            
        Returns:
            선택된 행동
        """
        # TODO: Epsilon-greedy 정책으로 행동을 선택하세요
        # 힌트 1: training이 True이고 random.random() < epsilon이면 무작위 행동 선택 (탐험)
        # 힌트 2: 그렇지 않으면 main_network로 Q-값을 계산하여 최대값의 행동 선택 (활용)
        # 힌트 3: 추론 시에는 torch.no_grad()를 사용하여 그래디언트 계산을 방지합니다
        # 힌트 4: 텐서를 device로 이동시키고, argmax()로 최대 Q-값의 인덱스를 가져옵니다
        #YOUR CODE HERE
        raise NotImplementedError("Epsilon-greedy action selection을 구현하세요")
            
    def store_transition(self, state, action, reward, next_state, done):
        """Replay buffer에 전이 저장"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update(self) -> Optional[float]:
        """
        Replay buffer에서 배치를 사용하여 네트워크 업데이트
        
        Returns:
            업데이트가 수행되면 손실 값, 아니면 None
        """
        if len(self.replay_buffer) < HYPERPARAMETERS['batch_size']:
            return None
            
        # 배치 샘플링
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(HYPERPARAMETERS['batch_size'])
            
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # TODO: DQN 학습 업데이트를 구현하세요 (Bellman Equation)
        # 힌트 1: main_network로 현재 상태의 Q-값을 계산하고 gather로 선택한 행동의 Q-값 추출
        # 힌트 2: target_network로 다음 상태의 최대 Q-값을 계산 (torch.no_grad() 사용)
        # 힌트 3: Target = reward + gamma * max Q(next_state) * (에피소드가 끝나지 않았으면)
        # 힌트 4: Loss = smooth_l1_loss(current_Q, target)로 손실 계산
        # 힌트 5: optimizer.zero_grad() → loss.backward() → clip_grad_norm → optimizer.step()
        # 힌트 6: 일정 스텝마다 target network를 main network로 업데이트
        #YOUR CODE HERE
        raise NotImplementedError("DQN update를 구현하세요")
        
    def update_target_network(self):
        """Main network의 가중치로 target network 업데이트"""
        self.target_network.load_state_dict(self.main_network.state_dict())
        
    def update_epsilon(self):
        """다음 에피소드를 위한 epsilon 업데이트"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
        
    def save_model(self, filepath: str):
        """모델 상태 저장"""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, filepath)
        
    def load_model(self, filepath: str):
        """모델 상태 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']


# ============================================================================
# 학습 관리자
# ============================================================================

class Trainer:
    """전체 학습 프로세스 관리"""
    
    def __init__(self, render: bool = False, load_model: Optional[str] = None):
        """
        Trainer 초기화
        
        Args:
            render: 환경 렌더링 여부
            load_model: 기존 모델 로드 경로
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {self.device}")
        
        # 환경 및 에이전트 초기화
        render_mode = "human" if render else None
        self.env = CarRacingWrapper(render_mode=render_mode)
        self.agent = DQNAgent(self.device)
        
        # 모델 로드 (지정된 경우)
        if load_model and os.path.exists(load_model):
            self.agent.load_model(load_model)
            print(f"모델 로드 완료: {load_model}")
            
        # 학습 통계
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_lengths = []
        
        # 디렉토리 생성
        self.models_dir = Path(__file__).parent.parent / "models" / "saved_weights"
        self.logs_dir = Path(__file__).parent.parent / "logs"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self, num_episodes: int):
        """
        지정된 에피소드 수만큼 에이전트 학습
        
        Args:
            num_episodes: 학습할 에피소드 수
        """
        print(f"{num_episodes} 에피소드 학습 시작...")
        print(f"하이퍼파라미터: {HYPERPARAMETERS}")
        print("-" * 60)
        
        start_time = time.time()
        best_reward = float('-inf')
        
        try:
            for episode in tqdm(range(num_episodes), desc="학습 중"):
                episode_reward, episode_loss, episode_length = self._train_episode()
                
                # 통계 업데이트
                self.episode_rewards.append(episode_reward)
                self.episode_losses.append(episode_loss)
                self.episode_lengths.append(episode_length)
                
                # 탐험 업데이트
                self.agent.update_epsilon()
                
                # 로깅
                if episode % HYPERPARAMETERS['log_interval'] == 0:
                    self._log_progress(episode, episode_reward, episode_loss)
                    
                # 모델 저장
                if episode % HYPERPARAMETERS['save_interval'] == 0:
                    model_path = self.models_dir / f"dqn_episode_{episode}.pth"
                    self.agent.save_model(str(model_path))
                    
                    # 최고 모델 저장
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        best_model_path = self.models_dir / "dqn_best.pth"
                        self.agent.save_model(str(best_model_path))
                        
        except KeyboardInterrupt:
            print("\n사용자에 의해 학습이 중단되었습니다")
            
        finally:
            # 최종 모델 저장
            final_model_path = self.models_dir / "dqn_final.pth"
            self.agent.save_model(str(final_model_path))
            
            # 학습 요약
            total_time = time.time() - start_time
            self._training_summary(total_time)
            
            # 결과 그래프
            self._plot_results()
            
            # 정리
            self.env.close()
            
    def _train_episode(self) -> Tuple[float, float, int]:
        """
        한 에피소드 학습
        
        Returns:
            (에피소드_보상, 평균_손실, 에피소드_길이) 튜플
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_losses = []
        step = 0
        
        for step in range(HYPERPARAMETERS['max_steps_per_episode']):
            # 행동 선택 및 수행
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # 전이 저장
            done = terminated or truncated
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # 에이전트 업데이트
            loss = self.agent.update()
            if loss is not None:
                episode_losses.append(loss)
                
            # 상태 및 보상 업데이트
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # 평균 손실 계산
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        return episode_reward, avg_loss, step + 1
        
    def _log_progress(self, episode: int, reward: float, loss: float):
        """학습 진행상황 로깅"""
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        avg_reward = np.mean(recent_rewards)
        
        print(f"에피소드 {episode:4d} | "
              f"보상: {reward:8.2f} | "
              f"평균 보상: {avg_reward:8.2f} | "
              f"손실: {loss:.4f} | "
              f"Epsilon: {self.agent.epsilon:.4f} | "
              f"버퍼: {len(self.agent.replay_buffer)}")
              
    def _training_summary(self, total_time: float):
        """학습 요약 출력"""
        print("\n" + "=" * 60)
        print("학습 요약")
        print("=" * 60)
        print(f"총 에피소드: {len(self.episode_rewards)}")
        print(f"총 시간: {total_time/60:.1f}분")
        print(f"평균 보상: {np.mean(self.episode_rewards):.2f}")
        print(f"최고 보상: {np.max(self.episode_rewards):.2f}")
        print(f"최종 epsilon: {self.agent.epsilon:.4f}")
        print(f"총 스텝 수: {self.agent.step_count}")
        
    def _plot_results(self):
        """학습 결과 그래프"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN 학습 결과')
        
        # 에피소드 보상
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('에피소드 보상')
        axes[0, 0].set_xlabel('에피소드')
        axes[0, 0].set_ylabel('보상')
        
        # 보상 이동평균
        window = 20
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'보상 이동평균 (윈도우={window})')
            axes[0, 1].set_xlabel('에피소드')
            axes[0, 1].set_ylabel('평균 보상')
            
        # 에피소드 손실
        axes[1, 0].plot(self.episode_losses)
        axes[1, 0].set_title('에피소드 손실')
        axes[1, 0].set_xlabel('에피소드')
        axes[1, 0].set_ylabel('손실')
        
        # 에피소드 길이
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('에피소드 길이')
        axes[1, 1].set_xlabel('에피소드')
        axes[1, 1].set_ylabel('스텝')
        
        plt.tight_layout()
        
        # 그래프 저장
        plot_path = self.logs_dir / "training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"학습 그래프 저장 완료: {plot_path}")
        
        plt.show()


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """DQN 학습을 실행하는 메인 함수"""
    parser = argparse.ArgumentParser(description='CarRacing을 위한 DQN 학습')
    parser.add_argument('--episodes', type=int, default=HYPERPARAMETERS['num_episodes'],
                       help='학습할 에피소드 수')
    parser.add_argument('--render', action='store_true',
                       help='학습 중 환경 렌더링')
    parser.add_argument('--load', type=str, default=None,
                       help='기존 모델 로드 경로')
    parser.add_argument('--seed', type=int, default=HYPERPARAMETERS['seed'],
                       help='랜덤 시드')
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("CarRacing 환경을 위한 DQN 학습")
    print("=" * 60)
    
    # Trainer 생성 및 학습 시작
    trainer = Trainer(render=args.render, load_model=args.load)
    trainer.train(args.episodes)
    
    print("\n학습이 성공적으로 완료되었습니다!")
    print("학습된 에이전트를 보려면 데모 스크립트를 실행하세요:")
    print("python games/demo_trained_agent.py")


if __name__ == "__main__":
    main()
