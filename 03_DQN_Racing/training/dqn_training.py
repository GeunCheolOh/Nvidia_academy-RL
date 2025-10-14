#!/usr/bin/env python3
"""
DQN Training Script for CarRacing Environment

This script implements a complete DQN (Deep Q-Networks) training pipeline
for the CarRacing-v2 environment. It includes all major DQN components:
- CNN-based Q-Network
- Experience Replay Buffer  
- Target Network
- Epsilon-Greedy Strategy
- Training Loop with Monitoring

Usage:
    python dqn_training.py [--episodes 500] [--render] [--load model.pth]

Author: DQN Racing Tutorial
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
# Hyperparameters Configuration
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
# Environment Preprocessing
# ============================================================================

class CarRacingWrapper:
    """Wrapper for CarRacing environment with preprocessing."""
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize CarRacing environment wrapper.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.env = gym.make('CarRacing-v3', render_mode=render_mode)
        self.frame_stack = HYPERPARAMETERS['frame_stack']
        self.image_size = HYPERPARAMETERS['image_size']
        
        # Frame buffer for stacking
        self.frames = deque(maxlen=self.frame_stack)
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial stacked frames."""
        obs, info = self.env.reset()
        
        # Preprocess initial frame
        processed_frame = self._preprocess_frame(obs)
        
        # Initialize frame stack with repeated first frame
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
            
        return self._get_stacked_frames()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action and return preprocessed observation.
        
        Args:
            action: Discrete action index
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert discrete action to continuous
        continuous_action = self._discrete_to_continuous(action)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        # Preprocess and stack frames
        processed_frame = self._preprocess_frame(obs)
        self.frames.append(processed_frame)
        stacked_frames = self._get_stacked_frames()
        
        return stacked_frames, reward, terminated, truncated, info
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame: resize, grayscale, normalize.
        
        Args:
            frame: Raw frame from environment
            
        Returns:
            Preprocessed frame
        """
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size
        resized_frame = cv2.resize(gray_frame, self.image_size)
        
        # Normalize to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        return normalized_frame
        
    def _get_stacked_frames(self) -> np.ndarray:
        """Get stacked frames as numpy array."""
        return np.array(list(self.frames))
        
    def _discrete_to_continuous(self, action: int) -> np.ndarray:
        """
        Convert discrete action to continuous action space.
        
        Args:
            action: Discrete action (0=left, 1=straight, 2=right, 3=brake)
            
        Returns:
            Continuous action [steering, gas, brake]
        """
        if action == 0:     # Turn left
            return np.array([-0.5, 0.3, 0.0])
        elif action == 1:   # Go straight
            return np.array([0.0, 0.5, 0.0])
        elif action == 2:   # Turn right
            return np.array([0.5, 0.3, 0.0])
        elif action == 3:   # Brake
            return np.array([0.0, 0.0, 0.8])
        else:
            return np.array([0.0, 0.0, 0.0])
            
    def close(self):
        """Close the environment."""
        self.env.close()


# ============================================================================
# DQN Network Architecture
# ============================================================================

class DQN(nn.Module):
    """CNN-based Deep Q-Network for CarRacing."""
    
    def __init__(self, action_dim: int = 4, input_channels: int = 4):
        """
        Initialize DQN network.
        
        Args:
            action_dim: Number of discrete actions
            input_channels: Number of input channels (frame stack)
        """
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Calculate conv output size
        self._conv_output_size = self._get_conv_output_size((input_channels, 84, 84))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate output size after conv layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self._forward_conv(dummy_input)
            return dummy_output.numel()
            
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv layers only."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action
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
    """Experience replay buffer for storing transitions."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of transitions."""
        # TODO: Replay Buffer에서 배치를 샘플링하세요
        # 힌트 1: random.sample을 사용하여 buffer에서 batch_size만큼 샘플링합니다
        # 힌트 2: zip(*batch)로 states, actions, rewards, next_states, dones를 분리합니다
        # 힌트 3: 각각을 적절한 torch 텐서로 변환합니다 (FloatTensor, LongTensor, BoolTensor)
        # 힌트 4: Experience Replay는 샘플 간 상관관계를 줄여 학습을 안정화합니다
        #YOUR CODE HERE
        raise NotImplementedError("Replay Buffer sampling을 구현하세요")
        
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


# ============================================================================
# DQN Agent
# ============================================================================

class DQNAgent:
    """DQN Agent with all training components."""
    
    def __init__(self, device: torch.device):
        """
        Initialize DQN agent.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        self.action_dim = 4  # left, straight, right, brake
        
        # Networks
        self.main_network = DQN(self.action_dim).to(device)
        self.target_network = DQN(self.action_dim).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.main_network.parameters(), 
            lr=HYPERPARAMETERS['learning_rate']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(HYPERPARAMETERS['buffer_size'])
        
        # Exploration strategy
        self.epsilon = HYPERPARAMETERS['epsilon_start']
        self.epsilon_decay = HYPERPARAMETERS['epsilon_decay']
        self.epsilon_min = HYPERPARAMETERS['epsilon_end']
        
        # Training counters
        self.step_count = 0
        self.episode_count = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # TODO: Epsilon-greedy 정책으로 행동을 선택하세요
        # 힌트 1: training이 True이고 random.random() < epsilon이면 무작위 행동 선택 (탐험)
        # 힌트 2: 그렇지 않으면 main_network로 Q-값을 계산하여 최대값의 행동 선택 (활용)
        # 힌트 3: 추론 시에는 torch.no_grad()를 사용하여 그래디언트 계산을 방지합니다
        # 힌트 4: 텐서를 device로 이동시키고, argmax()로 최대 Q-값의 인덱스를 가져옵니다
        #YOUR CODE HERE
        raise NotImplementedError("Epsilon-greedy action selection을 구현하세요")
            
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update(self) -> Optional[float]:
        """
        Update network using batch from replay buffer.
        
        Returns:
            Loss value if update performed, None otherwise
        """
        if len(self.replay_buffer) < HYPERPARAMETERS['batch_size']:
            return None
            
        # Sample batch
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
        """Update target network with main network weights."""
        self.target_network.load_state_dict(self.main_network.state_dict())
        
    def update_epsilon(self):
        """Update epsilon for next episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
        
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']


# ============================================================================
# Training Manager
# ============================================================================

class Trainer:
    """Manages the complete training process."""
    
    def __init__(self, render: bool = False, load_model: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            render: Whether to render environment
            load_model: Path to load existing model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize environment and agent
        render_mode = "human" if render else None
        self.env = CarRacingWrapper(render_mode=render_mode)
        self.agent = DQNAgent(self.device)
        
        # Load model if specified
        if load_model and os.path.exists(load_model):
            self.agent.load_model(load_model)
            print(f"Loaded model from {load_model}")
            
        # Training statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_lengths = []
        
        # Create directories
        self.models_dir = Path(__file__).parent.parent / "models" / "saved_weights"
        self.logs_dir = Path(__file__).parent.parent / "logs"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self, num_episodes: int):
        """
        Train the agent for specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Hyperparameters: {HYPERPARAMETERS}")
        print("-" * 60)
        
        start_time = time.time()
        best_reward = float('-inf')
        
        try:
            for episode in tqdm(range(num_episodes), desc="Training"):
                episode_reward, episode_loss, episode_length = self._train_episode()
                
                # Update statistics
                self.episode_rewards.append(episode_reward)
                self.episode_losses.append(episode_loss)
                self.episode_lengths.append(episode_length)
                
                # Update exploration
                self.agent.update_epsilon()
                
                # Logging
                if episode % HYPERPARAMETERS['log_interval'] == 0:
                    self._log_progress(episode, episode_reward, episode_loss)
                    
                # Save model
                if episode % HYPERPARAMETERS['save_interval'] == 0:
                    model_path = self.models_dir / f"dqn_episode_{episode}.pth"
                    self.agent.save_model(str(model_path))
                    
                    # Save best model
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        best_model_path = self.models_dir / "dqn_best.pth"
                        self.agent.save_model(str(best_model_path))
                        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            
        finally:
            # Save final model
            final_model_path = self.models_dir / "dqn_final.pth"
            self.agent.save_model(str(final_model_path))
            
            # Training summary
            total_time = time.time() - start_time
            self._training_summary(total_time)
            
            # Plot results
            self._plot_results()
            
            # Cleanup
            self.env.close()
            
    def _train_episode(self) -> Tuple[float, float, int]:
        """
        Train for one episode.
        
        Returns:
            Tuple of (episode_reward, average_loss, episode_length)
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_losses = []
        step = 0
        
        for step in range(HYPERPARAMETERS['max_steps_per_episode']):
            # Select and take action
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # Store transition
            done = terminated or truncated
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = self.agent.update()
            if loss is not None:
                episode_losses.append(loss)
                
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # Calculate average loss
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        return episode_reward, avg_loss, step + 1
        
    def _log_progress(self, episode: int, reward: float, loss: float):
        """Log training progress."""
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        avg_reward = np.mean(recent_rewards)
        
        print(f"Episode {episode:4d} | "
              f"Reward: {reward:8.2f} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Loss: {loss:.4f} | "
              f"Epsilon: {self.agent.epsilon:.4f} | "
              f"Buffer: {len(self.agent.replay_buffer)}")
              
    def _training_summary(self, total_time: float):
        """Print training summary."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Best reward: {np.max(self.episode_rewards):.2f}")
        print(f"Final epsilon: {self.agent.epsilon:.4f}")
        print(f"Total steps: {self.agent.step_count}")
        
    def _plot_results(self):
        """Plot training results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Results')
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Moving average of rewards
        window = 20
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            
        # Episode losses
        axes[1, 0].plot(self.episode_losses)
        axes[1, 0].set_title('Episode Losses')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        
        # Episode lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.logs_dir / "training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {plot_path}")
        
        plt.show()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to run DQN training."""
    parser = argparse.ArgumentParser(description='DQN Training for CarRacing')
    parser.add_argument('--episodes', type=int, default=HYPERPARAMETERS['num_episodes'],
                       help='Number of episodes to train')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load existing model')
    parser.add_argument('--seed', type=int, default=HYPERPARAMETERS['seed'],
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("🏎️  DQN Training for CarRacing Environment")
    print("=" * 60)
    
    # Create trainer and start training
    trainer = Trainer(render=args.render, load_model=args.load)
    trainer.train(args.episodes)
    
    print("\n🎉 Training completed successfully!")
    print("Run demo script to see the trained agent in action:")
    print("python games/demo_trained_agent.py")


if __name__ == "__main__":
    main()