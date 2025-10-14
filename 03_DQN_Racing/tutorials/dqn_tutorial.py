#!/usr/bin/env python3
"""
DQN (Deep Q-Networks) Tutorial

This tutorial provides a comprehensive walkthrough of Deep Q-Networks (DQN) 
for reinforcement learning. Each section builds upon the previous one to 
create a complete understanding of DQN components.

Sections:
1. Q-Learning Review and Limitations
2. Neural Network Architecture for DQN
3. Experience Replay Buffer
4. Target Network Mechanism
5. Epsilon-Greedy Strategy
6. Loss Function and Optimization

Author: DQN Racing Tutorial
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from collections import deque
import cv2
from typing import Tuple, List, Optional
import time


# ============================================================================
# Section 1: Q-Learning Review and Limitations
# ============================================================================

class QLearningDemo:
    """Demonstrates Q-Learning limitations with large state spaces."""
    
    def __init__(self):
        self.q_table_size_examples = {
            "GridWorld 4x4": 16,
            "GridWorld 10x10": 100,
            "Atari (84x84 grayscale)": 256**(84*84),
            "CarRacing (84x84x3)": 256**(84*84*3)
        }
        
    def demonstrate_state_space_explosion(self):
        """Show how state space grows exponentially."""
        print("=" * 60)
        print("SECTION 1: Q-LEARNING LIMITATIONS")
        print("=" * 60)
        
        print("Q-Table size for different environments:")
        print("-" * 40)
        
        for env_name, size in self.q_table_size_examples.items():
            if size < 1e6:
                print(f"{env_name:<25}: {size:,} states")
            else:
                print(f"{env_name:<25}: {size:.2e} states (IMPOSSIBLE!)")
                
        print("\nKey Insight:")
        print("📊 Q-Tables work for small, discrete state spaces")
        print("❌ Q-Tables fail for large, continuous, or high-dimensional states")
        print("✅ Neural Networks can approximate Q-functions for complex states")
        print()


# ============================================================================
# Section 2: Neural Network Architecture for DQN
# ============================================================================

class DQNNetwork(nn.Module):
    """CNN-based Deep Q-Network for CarRacing environment."""
    
    def __init__(self, action_dim: int = 3, input_channels: int = 4):
        """
        Initialize DQN network.
        
        Args:
            action_dim: Number of possible actions (steering, gas, brake)
            input_channels: Number of input channels (frame stack)
        """
        super(DQNNetwork, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size of flattened features
        self._conv_output_size = self._get_conv_output_size((input_channels, 84, 84))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate the output size after convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self._forward_conv(dummy_input)
            return dummy_output.numel()
            
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers only."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action
        """
        # Convolutional feature extraction
        x = self._forward_conv(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
    def get_network_info(self):
        """Get information about the network architecture."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            "total_parameters": total_params,
            "conv_output_size": self._conv_output_size,
            "input_shape": (4, 84, 84),
            "output_shape": 3
        }
        return info


def demonstrate_network_architecture():
    """Demonstrate the DQN network architecture."""
    print("=" * 60)
    print("SECTION 2: NEURAL NETWORK ARCHITECTURE")
    print("=" * 60)
    
    # Create network
    dqn = DQNNetwork(action_dim=3, input_channels=4)
    info = dqn.get_network_info()
    
    print("DQN Network Architecture:")
    print("-" * 30)
    print(f"Input: {info['input_shape']} (4 stacked frames of 84x84)")
    print("Conv1: 32 filters, 8x8 kernel, stride 4")
    print("Conv2: 64 filters, 4x4 kernel, stride 2") 
    print("Conv3: 64 filters, 3x3 kernel, stride 1")
    print(f"Flattened features: {info['conv_output_size']}")
    print("FC1: 512 neurons")
    print(f"Output: {info['output_shape']} Q-values")
    print(f"\nTotal parameters: {info['total_parameters']:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 4, 84, 84)
    with torch.no_grad():
        output = dqn(dummy_input)
        
    print(f"\nExample forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output Q-values: {output.squeeze().numpy()}")
    print()


# ============================================================================
# Section 3: Experience Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions as separate tensors
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} samples, need {batch_size}")
            
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
        
    def get_statistics(self):
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {"size": 0, "capacity": self.capacity, "utilization": 0.0}
            
        rewards = [transition[2] for transition in self.buffer]
        
        stats = {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards)
        }
        return stats


def demonstrate_experience_replay():
    """Demonstrate experience replay functionality."""
    print("=" * 60)
    print("SECTION 3: EXPERIENCE REPLAY BUFFER")
    print("=" * 60)
    
    # Create buffer
    buffer = ReplayBuffer(capacity=1000)
    
    print("Adding sample experiences to buffer...")
    
    # Add some dummy experiences
    for i in range(150):
        state = np.random.random((4, 84, 84))
        action = np.random.randint(0, 3)
        reward = np.random.normal(0, 1)  # Random reward
        next_state = np.random.random((4, 84, 84))
        done = np.random.random() < 0.1  # 10% chance of episode end
        
        buffer.push(state, action, reward, next_state, done)
        
    # Show buffer statistics
    stats = buffer.get_statistics()
    print(f"Buffer Statistics:")
    print(f"  Size: {stats['size']}/{stats['capacity']}")
    print(f"  Utilization: {stats['utilization']:.1%}")
    print(f"  Average reward: {stats['avg_reward']:.3f}")
    print(f"  Reward std: {stats['reward_std']:.3f}")
    
    # Demonstrate sampling
    print(f"\nSampling batch of 32 experiences...")
    states, actions, rewards, next_states, dones = buffer.sample(32)
    
    print(f"Batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Next states: {next_states.shape}")
    print(f"  Dones: {dones.shape}")
    
    print("\nBenefits of Experience Replay:")
    print("✅ Breaks correlation between consecutive experiences")
    print("✅ Enables multiple learning updates from same experience")
    print("✅ Improves sample efficiency")
    print("✅ Stabilizes training by mixing old and new experiences")
    print()


# ============================================================================
# Section 4: Target Network Mechanism
# ============================================================================

class TargetNetworkDemo:
    """Demonstrates the target network concept."""
    
    def __init__(self):
        self.main_network = DQNNetwork()
        self.target_network = DQNNetwork()
        
        # Copy main network weights to target network
        self.hard_update()
        
    def hard_update(self):
        """Copy main network weights to target network."""
        self.target_network.load_state_dict(self.main_network.state_dict())
        
    def soft_update(self, tau: float = 0.001):
        """
        Soft update target network weights.
        
        Args:
            tau: Soft update parameter (0 = no update, 1 = hard update)
        """
        for target_param, main_param in zip(
            self.target_network.parameters(), 
            self.main_network.parameters()
        ):
            target_param.data.copy_(
                tau * main_param.data + (1.0 - tau) * target_param.data
            )
            
    def compare_networks(self) -> float:
        """Compare parameter differences between networks."""
        total_diff = 0.0
        total_params = 0
        
        for target_param, main_param in zip(
            self.target_network.parameters(),
            self.main_network.parameters()
        ):
            diff = torch.norm(target_param - main_param).item()
            total_diff += diff
            total_params += target_param.numel()
            
        return total_diff / total_params


def demonstrate_target_network():
    """Demonstrate target network mechanism."""
    print("=" * 60)
    print("SECTION 4: TARGET NETWORK MECHANISM")
    print("=" * 60)
    
    demo = TargetNetworkDemo()
    
    print("Target Network Concept:")
    print("-" * 25)
    print("Main Network:   Used for action selection and learning")
    print("Target Network: Used for Q-target calculation (stable)")
    print()
    
    # Show initial state
    initial_diff = demo.compare_networks()
    print(f"Initial parameter difference: {initial_diff:.6f}")
    
    # Simulate training updates to main network
    optimizer = optim.Adam(demo.main_network.parameters(), lr=0.001)
    
    print("\nSimulating training updates...")
    differences = []
    
    for step in range(20):
        # Dummy loss to update main network
        dummy_input = torch.randn(1, 4, 84, 84)
        loss = demo.main_network(dummy_input).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure difference
        diff = demo.compare_networks()
        differences.append(diff)
        
        if step % 5 == 0:
            print(f"Step {step:2d}: Parameter difference = {diff:.6f}")
            
        # Hard update every 10 steps
        if step == 10:
            demo.hard_update()
            print(f"        Hard update performed!")
            
    print("\nTarget Network Benefits:")
    print("✅ Prevents moving target problem in Q-learning")
    print("✅ Stabilizes training by providing consistent targets")
    print("✅ Reduces correlation between Q-values and targets")
    print("✅ Hard updates every N steps maintain stability")
    print()


# ============================================================================
# Section 5: Epsilon-Greedy Strategy
# ============================================================================

class EpsilonGreedyStrategy:
    """Implements epsilon-greedy exploration strategy."""
    
    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                 epsilon_decay: float = 0.995):
        """
        Initialize epsilon-greedy strategy.
        
        Args:
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay factor per episode
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.episode = 0
        
    def get_action(self, q_values: torch.Tensor) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            q_values: Q-values for all actions
            
        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(len(q_values))
        else:
            # Exploit: best action
            return q_values.argmax().item()
            
    def update_epsilon(self):
        """Update epsilon for next episode."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        self.episode += 1
        
    def get_epsilon_schedule(self, num_episodes: int) -> List[float]:
        """Get epsilon values for given number of episodes."""
        epsilons = []
        epsilon = self.epsilon_start
        
        for _ in range(num_episodes):
            epsilons.append(epsilon)
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            
        return epsilons


def demonstrate_epsilon_greedy():
    """Demonstrate epsilon-greedy exploration strategy."""
    print("=" * 60)
    print("SECTION 5: EPSILON-GREEDY STRATEGY")
    print("=" * 60)
    
    strategy = EpsilonGreedyStrategy(
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Get epsilon schedule
    num_episodes = 500
    epsilon_schedule = strategy.get_epsilon_schedule(num_episodes)
    
    # Show key points
    print("Epsilon Decay Schedule:")
    print("-" * 22)
    milestones = [0, 50, 100, 200, 300, 400, 499]
    for ep in milestones:
        print(f"Episode {ep:3d}: ε = {epsilon_schedule[ep]:.4f}")
        
    # Simulate action selection
    print(f"\nAction Selection Simulation (Episode 0, ε = {epsilon_schedule[0]:.3f}):")
    dummy_q_values = torch.tensor([0.1, 0.8, 0.3])  # [steering, gas, brake]
    
    actions = []
    for _ in range(100):
        action = strategy.get_action(dummy_q_values)
        actions.append(action)
        
    action_counts = np.bincount(actions, minlength=3)
    action_names = ['Steering', 'Gas', 'Brake']
    
    print("Action distribution (100 selections):")
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        percentage = count / 100 * 100
        print(f"  {name}: {count:2d}/100 ({percentage:4.1f}%)")
        
    print(f"\nNote: Gas has highest Q-value ({dummy_q_values[1]:.1f}) but random")
    print("exploration still selects other actions frequently.")
    
    print("\nExploration vs Exploitation Trade-off:")
    print("✅ High ε: More exploration, discovers new strategies")
    print("✅ Low ε:  More exploitation, uses learned knowledge")
    print("✅ Decay:  Gradually shift from exploration to exploitation")
    print()


# ============================================================================
# Section 6: Loss Function and Optimization
# ============================================================================

def demonstrate_loss_function():
    """Demonstrate DQN loss function and optimization."""
    print("=" * 60)
    print("SECTION 6: LOSS FUNCTION AND OPTIMIZATION")
    print("=" * 60)
    
    print("DQN Loss Function (Temporal Difference Error):")
    print("-" * 48)
    print("Target: y = r + γ * max(Q_target(s', a'))")
    print("Loss:   L = Huber(Q_main(s, a) - y)")
    print()
    
    # Create networks
    main_network = DQNNetwork()
    target_network = DQNNetwork()
    target_network.load_state_dict(main_network.state_dict())
    
    # Dummy batch
    batch_size = 4
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.LongTensor([0, 1, 2, 1])  # [steering, gas, brake, gas]
    rewards = torch.FloatTensor([0.1, 1.0, -0.5, 0.8])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.BoolTensor([False, False, True, False])
    gamma = 0.99
    
    print("Example Batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Actions: {actions.tolist()}")
    print(f"  Rewards: {rewards.tolist()}")
    print(f"  Dones: {dones.tolist()}")
    print()
    
    # Forward pass
    with torch.no_grad():
        # Current Q-values
        current_q_values = main_network(states)
        current_q_values_selected = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        next_q_values = target_network(next_states)
        next_q_values_max = next_q_values.max(1)[0]
        
        # Compute targets
        targets = rewards + (gamma * next_q_values_max * (~dones))
        
    print("Q-Value Computation:")
    print(f"  Current Q-values shape: {current_q_values.shape}")
    print(f"  Selected Q-values: {current_q_values_selected.squeeze().detach().numpy()}")
    print(f"  Next max Q-values: {next_q_values_max.detach().numpy()}")
    print(f"  Targets: {targets.detach().numpy()}")
    
    # Compute loss
    td_errors = current_q_values_selected.squeeze() - targets
    huber_loss = F.smooth_l1_loss(current_q_values_selected.squeeze(), targets)
    mse_loss = F.mse_loss(current_q_values_selected.squeeze(), targets)
    
    print(f"\nLoss Computation:")
    print(f"  TD errors: {td_errors.detach().numpy()}")
    print(f"  Huber loss: {huber_loss.item():.4f}")
    print(f"  MSE loss: {mse_loss.item():.4f}")
    
    print("\nWhy Huber Loss?")
    print("✅ Less sensitive to outliers than MSE")
    print("✅ Provides stable gradients for large errors")
    print("✅ Behaves like MSE for small errors, MAE for large errors")
    print("✅ Improves training stability")
    print()


# ============================================================================
# Main Tutorial Runner
# ============================================================================

def run_complete_tutorial():
    """Run the complete DQN tutorial."""
    print("🎯 DQN (Deep Q-Networks) Tutorial")
    print("=" * 60)
    print("This tutorial covers all essential components of DQN")
    print("for reinforcement learning in the CarRacing environment.")
    print("=" * 60)
    print()
    
    # Section 1: Q-Learning limitations
    demo = QLearningDemo()
    demo.demonstrate_state_space_explosion()
    input("Press Enter to continue to Section 2...")
    print()
    
    # Section 2: Network architecture
    demonstrate_network_architecture()
    input("Press Enter to continue to Section 3...")
    print()
    
    # Section 3: Experience replay
    demonstrate_experience_replay()
    input("Press Enter to continue to Section 4...")
    print()
    
    # Section 4: Target network
    demonstrate_target_network()
    input("Press Enter to continue to Section 5...")
    print()
    
    # Section 5: Epsilon-greedy
    demonstrate_epsilon_greedy()
    input("Press Enter to continue to Section 6...")
    print()
    
    # Section 6: Loss function
    demonstrate_loss_function()
    
    print("=" * 60)
    print("🎉 Tutorial Complete!")
    print("=" * 60)
    print("You now understand the key components of DQN:")
    print("✅ Neural networks for Q-function approximation")
    print("✅ Experience replay for stable learning")
    print("✅ Target networks for training stability")
    print("✅ Epsilon-greedy for exploration")
    print("✅ Huber loss for robust optimization")
    print()
    print("Next step: Run the training script to see DQN in action!")
    print("Command: python training/dqn_training.py")


if __name__ == "__main__":
    try:
        run_complete_tutorial()
    except KeyboardInterrupt:
        print("\n\nTutorial interrupted by user. Thanks for learning! 📚")
    except Exception as e:
        print(f"\nError in tutorial: {e}")
        print("Please check your environment setup.")