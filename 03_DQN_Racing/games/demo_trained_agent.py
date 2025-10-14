#!/usr/bin/env python3
"""
Trained DQN Agent Demonstration

This script demonstrates a trained DQN agent playing CarRacing.
It loads saved model weights and shows the agent's performance
compared to random actions.

Usage:
    python demo_trained_agent.py [--model model.pth] [--episodes 5] [--compare]

Author: DQN Racing Tutorial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import cv2
import argparse
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List, Dict
import pygame


# Import from training script
import sys
sys.path.append(str(Path(__file__).parent.parent / "training"))
from dqn_training import DQN, CarRacingWrapper, HYPERPARAMETERS


# ============================================================================
# Agent Demo Class
# ============================================================================

class DQNDemo:
    """Demonstration class for trained DQN agents."""
    
    def __init__(self, model_path: Optional[str] = None, render: bool = True):
        """
        Initialize demo environment.
        
        Args:
            model_path: Path to trained model weights
            render: Whether to render the environment
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize environment
        render_mode = "human" if render else "rgb_array"
        self.env = CarRacingWrapper(render_mode=render_mode)
        
        # Initialize network
        self.network = DQN(action_dim=4).to(self.device)
        
        # Load model if provided
        self.model_loaded = False
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._find_and_load_best_model()
            
        # Demo statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = []
        
    def load_model(self, model_path: str):
        """Load trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'main_network' in checkpoint:
                state_dict = checkpoint['main_network']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            self.network.load_state_dict(state_dict)
            self.network.eval()  # Set to evaluation mode
            
            print(f"✓ Model loaded from: {model_path}")
            self.model_loaded = True
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("Will use random agent instead")
            self.model_loaded = False
            
    def _find_and_load_best_model(self):
        """Find and load the best available model."""
        models_dir = Path(__file__).parent.parent / "models" / "saved_weights"
        
        # Try to find best model
        best_model = models_dir / "dqn_best.pth"
        if best_model.exists():
            self.load_model(str(best_model))
            return
            
        # Try to find final model
        final_model = models_dir / "dqn_final.pth"
        if final_model.exists():
            self.load_model(str(final_model))
            return
            
        # Try to find any model
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            if model_files:
                # Sort by modification time and take the latest
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                self.load_model(str(latest_model))
                return
                
        print("⚠️  No trained models found!")
        print("Please train a model first by running:")
        print("python training/dqn_training.py")
        
    def select_action(self, state: np.ndarray, use_model: bool = True) -> int:
        """
        Select action using trained model or random policy.
        
        Args:
            state: Current state
            use_model: Whether to use trained model or random policy
            
        Returns:
            Selected action
        """
        if use_model and self.model_loaded:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.network(state_tensor)
                return q_values.argmax().item()
        else:
            # Random action
            return np.random.randint(0, 4)
            
    def run_episode(self, use_model: bool = True, max_steps: int = 1000) -> Tuple[float, int, List[int]]:
        """
        Run single episode with agent.
        
        Args:
            use_model: Whether to use trained model
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (total_reward, episode_length, actions_taken)
        """
        state = self.env.reset()
        total_reward = 0.0
        actions_taken = []
        
        for step in range(max_steps):
            action = self.select_action(state, use_model)
            actions_taken.append(action)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
                
        return total_reward, step + 1, actions_taken
        
    def demo_single_agent(self, num_episodes: int = 5, use_model: bool = True):
        """
        Demonstrate single agent for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            use_model: Whether to use trained model
        """
        agent_type = "Trained DQN" if (use_model and self.model_loaded) else "Random"
        print(f"\n{'='*60}")
        print(f"DEMONSTRATING {agent_type.upper()} AGENT")
        print(f"{'='*60}")
        
        episode_rewards = []
        episode_lengths = []
        all_actions = []
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print("-" * 30)
            
            start_time = time.time()
            reward, length, actions = self.run_episode(use_model)
            episode_time = time.time() - start_time
            
            episode_rewards.append(reward)
            episode_lengths.append(length)
            all_actions.extend(actions)
            
            print(f"Reward: {reward:8.2f}")
            print(f"Length: {length:4d} steps")
            print(f"Time:   {episode_time:6.2f} seconds")
            
            # Action distribution for this episode
            action_counts = np.bincount(actions, minlength=4)
            action_names = ['Left', 'Straight', 'Right', 'Brake']
            print("Actions:", end="")
            for i, (name, count) in enumerate(zip(action_names, action_counts)):
                print(f" {name}: {count:3d}", end="")
            print()
            
        # Summary statistics
        print(f"\n{agent_type} Agent Summary:")
        print("-" * 30)
        print(f"Episodes:      {num_episodes}")
        print(f"Avg Reward:    {np.mean(episode_rewards):8.2f}")
        print(f"Std Reward:    {np.std(episode_rewards):8.2f}")
        print(f"Best Reward:   {np.max(episode_rewards):8.2f}")
        print(f"Worst Reward:  {np.min(episode_rewards):8.2f}")
        print(f"Avg Length:    {np.mean(episode_lengths):6.1f} steps")
        
        # Overall action distribution
        total_action_counts = np.bincount(all_actions, minlength=4)
        print(f"Action Distribution:")
        for i, (name, count) in enumerate(zip(action_names, total_action_counts)):
            percentage = count / len(all_actions) * 100
            print(f"  {name:8}: {count:5d} ({percentage:5.1f}%)")
            
        return episode_rewards, episode_lengths
        
    def compare_agents(self, num_episodes: int = 5):
        """
        Compare trained agent vs random agent.
        
        Args:
            num_episodes: Number of episodes per agent
        """
        if not self.model_loaded:
            print("⚠️  No trained model available for comparison!")
            print("Running random agent demonstration only...")
            self.demo_single_agent(num_episodes, use_model=False)
            return
            
        print(f"\n{'='*60}")
        print("AGENT COMPARISON")
        print(f"{'='*60}")
        
        # Run trained agent
        print(f"\n🤖 Testing Trained DQN Agent...")
        trained_rewards, trained_lengths = self.demo_single_agent(num_episodes, use_model=True)
        
        # Run random agent
        print(f"\n🎲 Testing Random Agent...")
        random_rewards, random_lengths = self.demo_single_agent(num_episodes, use_model=False)
        
        # Statistical comparison
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        
        print(f"{'Metric':<20} {'Trained DQN':<15} {'Random':<15} {'Improvement':<15}")
        print("-" * 65)
        
        # Rewards
        trained_avg = np.mean(trained_rewards)
        random_avg = np.mean(random_rewards)
        reward_improvement = ((trained_avg - random_avg) / abs(random_avg)) * 100
        
        print(f"{'Avg Reward':<20} {trained_avg:<15.2f} {random_avg:<15.2f} {reward_improvement:<15.1f}%")
        
        # Episode lengths
        trained_len_avg = np.mean(trained_lengths)
        random_len_avg = np.mean(random_lengths)
        length_improvement = ((trained_len_avg - random_len_avg) / random_len_avg) * 100
        
        print(f"{'Avg Length':<20} {trained_len_avg:<15.1f} {random_len_avg:<15.1f} {length_improvement:<15.1f}%")
        
        # Best performance
        trained_best = np.max(trained_rewards)
        random_best = np.max(random_rewards)
        best_improvement = ((trained_best - random_best) / abs(random_best)) * 100
        
        print(f"{'Best Reward':<20} {trained_best:<15.2f} {random_best:<15.2f} {best_improvement:<15.1f}%")
        
        # Consistency (lower std is better)
        trained_std = np.std(trained_rewards)
        random_std = np.std(random_rewards)
        consistency_improvement = ((random_std - trained_std) / random_std) * 100
        
        print(f"{'Consistency':<20} {trained_std:<15.2f} {random_std:<15.2f} {consistency_improvement:<15.1f}%")
        
        # Statistical significance test
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_ind(trained_rewards, random_rewards)
            print(f"\nStatistical Test (t-test):")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value:     {p_value:.4f}")
            if p_value < 0.05:
                print("  Result: Statistically significant difference! 🎉")
            else:
                print("  Result: No significant difference (need more training)")
        except ImportError:
            print("\nInstall scipy for statistical significance testing")
            
        # Create comparison plot
        self._plot_comparison(trained_rewards, random_rewards)
        
    def _plot_comparison(self, trained_rewards: List[float], random_rewards: List[float]):
        """Create comparison plots."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Episode rewards comparison
            episodes = range(1, len(trained_rewards) + 1)
            ax1.plot(episodes, trained_rewards, 'b-o', label='Trained DQN', linewidth=2)
            ax1.plot(episodes, random_rewards, 'r-s', label='Random', linewidth=2)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot comparison
            ax2.boxplot([trained_rewards, random_rewards], 
                       labels=['Trained DQN', 'Random'])
            ax2.set_ylabel('Reward')
            ax2.set_title('Reward Distribution Comparison')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            logs_dir = Path(__file__).parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            plot_path = logs_dir / "agent_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison plot saved to: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Could not create plots: {e}")
            
    def interactive_demo(self):
        """Interactive demonstration with user controls."""
        if not self.model_loaded:
            print("⚠️  No trained model available!")
            return
            
        print(f"\n{'='*60}")
        print("INTERACTIVE DEMO")
        print(f"{'='*60}")
        print("Controls:")
        print("  SPACE - Toggle between Trained/Random agent")
        print("  R     - Reset episode")
        print("  ESC   - Quit")
        print("  P     - Pause/Resume")
        print("-" * 60)
        
        pygame.init()
        clock = pygame.time.Clock()
        
        use_model = True
        paused = False
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        use_model = not use_model
                        agent_type = "Trained DQN" if use_model else "Random"
                        print(f"Switched to: {agent_type}")
                    elif event.key == pygame.K_r:
                        state = self.env.reset()
                        episode_reward = 0.0
                        episode_steps = 0
                        print("Episode reset")
                    elif event.key == pygame.K_p:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                        
            if not paused:
                action = self.select_action(state, use_model)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Display info
                agent_type = "DQN" if use_model else "RND"
                action_names = ['LEFT', 'STRAIGHT', 'RIGHT', 'BRAKE']
                print(f"\r{agent_type} | Steps: {episode_steps:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Action: {action_names[action]}", end="", flush=True)
                
                if terminated or truncated:
                    print(f"\nEpisode ended! Final reward: {episode_reward:.2f}")
                    state = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
                    
            clock.tick(30)  # 30 FPS
            
        pygame.quit()
        print("\nInteractive demo ended")
        
    def cleanup(self):
        """Clean up resources."""
        self.env.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function for agent demonstration."""
    parser = argparse.ArgumentParser(description='DQN Agent Demonstration')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--compare', action='store_true',
                       help='Compare trained vs random agent')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive demo')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    
    args = parser.parse_args()
    
    print("🏎️  DQN Agent Demonstration")
    print("=" * 60)
    
    # Create demo
    demo = DQNDemo(model_path=args.model, render=not args.no_render)
    
    try:
        if args.interactive:
            demo.interactive_demo()
        elif args.compare:
            demo.compare_agents(args.episodes)
        else:
            demo.demo_single_agent(args.episodes, use_model=True)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        demo.cleanup()
        
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()