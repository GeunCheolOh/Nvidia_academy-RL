#!/usr/bin/env python3
"""
Manual CarRacing Game Test

This script allows you to manually play the CarRacing game using keyboard controls.
It's useful for understanding the game environment before implementing AI agents.

Controls:
    Arrow Keys:
        ↑ (Up)    - Accelerate
        ↓ (Down)  - Brake
        ← (Left)  - Steer Left
        → (Right) - Steer Right
    
    Other Keys:
        ESC       - Quit game
        R         - Reset episode
        SPACE     - Pause/Resume

Author: DQN Racing Tutorial
"""

import gymnasium as gym
import pygame
import numpy as np
import sys
import time
from typing import Tuple, Dict, Any


class ManualCarRacing:
    """Manual control interface for CarRacing environment."""
    
    def __init__(self, render_mode: str = "human"):
        """
        Initialize the manual racing environment.
        
        Args:
            render_mode: Rendering mode for the environment
        """
        self.env = None
        self.render_mode = render_mode
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.paused = False
        
        # Game statistics
        self.episode_count = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.max_reward = float('-inf')
        self.episode_rewards = []
        
        # Control state
        self.action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake] or discrete action
        self.keys_pressed = set()
        
        print("Manual CarRacing Game")
        print("=" * 40)
        print("Controls:")
        print("  ↑ (Up)    - Accelerate")
        print("  ↓ (Down)  - Brake") 
        print("  ← (Left)  - Steer Left")
        print("  → (Right) - Steer Right")
        print("  ESC       - Quit")
        print("  R         - Reset")
        print("  SPACE     - Pause/Resume")
        print("=" * 40)
        
    def init_environment(self):
        """Initialize the racing environment (CarRacing or fallback)."""
        # Try CarRacing first
        try:
            self.env = gym.make('CarRacing-v3', render_mode=self.render_mode)
            self.env_name = "CarRacing-v3"
            print("✓ CarRacing environment initialized successfully")
            return True
        except Exception as e:
            print(f"⚠️  CarRacing not available: {e}")
            
        # Fallback to CartPole if CarRacing still fails
        try:
            self.env = gym.make('CartPole-v1', render_mode=self.render_mode)
            self.env_name = "CartPole-v1"
            print("✓ Using CartPole as fallback environment")
            print("  (CarRacing requires Box2D: pip install 'gymnasium[box2d]')")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize any environment: {e}")
            return False
            
    def reset_episode(self):
        """Reset the environment for a new episode."""
        if self.env is None:
            return None
            
        try:
            obs, info = self.env.reset()
            
            # Update statistics
            if self.total_reward > 0:
                self.episode_rewards.append(self.total_reward)
                if self.total_reward > self.max_reward:
                    self.max_reward = self.total_reward
                    
            self.episode_count += 1
            self.total_reward = 0.0
            self.step_count = 0
            self.action = np.array([0.0, 0.0, 0.0])
            
            print(f"\n--- Episode {self.episode_count} Started ---")
            return obs
            
        except Exception as e:
            print(f"Error resetting environment: {e}")
            return None
            
    def process_keyboard_input(self) -> bool:
        """
        Process keyboard input and update action.
        
        Returns:
            bool: False if quit requested, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Game Paused" if self.paused else "Game Resumed")
                    
            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)
                
        # Update action based on currently pressed keys
        self.update_action_from_keys()
        return True
        
    def update_action_from_keys(self):
        """Update action array based on currently pressed keys."""
        if hasattr(self, 'env_name') and 'CartPole' in self.env_name:
            # CartPole: discrete actions (0=left, 1=right)
            self.action = 0  # Default: push left
            
            if pygame.K_LEFT in self.keys_pressed:
                self.action = 0  # Push cart left
            elif pygame.K_RIGHT in self.keys_pressed:
                self.action = 1  # Push cart right
        else:
            # CarRacing: continuous actions [steering, gas, brake]
            self.action = np.array([0.0, 0.0, 0.0])
            
            # Steering (left/right)
            if pygame.K_LEFT in self.keys_pressed:
                self.action[0] = -1.0  # Steer left
            elif pygame.K_RIGHT in self.keys_pressed:
                self.action[0] = 1.0   # Steer right
                
            # Gas (accelerate)
            if pygame.K_UP in self.keys_pressed:
                self.action[1] = 1.0   # Gas
                
            # Brake
            if pygame.K_DOWN in self.keys_pressed:
                self.action[2] = 1.0   # Brake
            
    def display_info(self):
        """Display game information on console."""
        if self.step_count % 60 == 0:  # Update every second (60 FPS)
            if hasattr(self, 'env_name') and 'CartPole' in self.env_name:
                action_names = ['Push Left', 'Push Right']
                action_str = action_names[self.action] if self.action < len(action_names) else str(self.action)
            else:
                action_str = f"[{self.action[0]:.1f}, {self.action[1]:.1f}, {self.action[2]:.1f}]"
                
            info_str = (
                f"Episode: {self.episode_count} | "
                f"Step: {self.step_count} | "
                f"Reward: {self.total_reward:.1f} | "
                f"Action: {action_str}"
            )
            print(f"\r{info_str}", end="", flush=True)
            
    def display_statistics(self):
        """Display game statistics."""
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards)
            print(f"\n\nGame Statistics:")
            print(f"  Episodes completed: {len(self.episode_rewards)}")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Best reward: {self.max_reward:.2f}")
            print(f"  Last reward: {self.episode_rewards[-1]:.2f}")
        else:
            print(f"\nTotal steps: {self.step_count}")
            print(f"Current reward: {self.total_reward:.2f}")
            
    def run(self):
        """Run the manual racing game."""
        if not self.init_environment():
            return
            
        pygame.init()
        
        # Start first episode
        obs = self.reset_episode()
        if obs is None:
            return
            
        print("Game started! Use arrow keys to control the car.")
        running = True
        
        try:
            while running:
                # Process input
                running = self.process_keyboard_input()
                if not running:
                    break
                    
                # Skip game logic if paused
                if self.paused:
                    self.clock.tick(10)  # Lower FPS when paused
                    continue
                    
                # Take action in environment
                try:
                    obs, reward, terminated, truncated, info = self.env.step(self.action)
                    
                    # Update statistics
                    self.total_reward += reward
                    self.step_count += 1
                    
                    # Display info
                    self.display_info()
                    
                    # Check if episode ended
                    if terminated or truncated:
                        print(f"\nEpisode {self.episode_count} ended!")
                        print(f"Final reward: {self.total_reward:.2f}")
                        print("Press 'R' to reset or ESC to quit")
                        
                except Exception as e:
                    print(f"\nError during step: {e}")
                    break
                    
                # Control FPS
                self.clock.tick(self.fps)
                
        except KeyboardInterrupt:
            print("\nGame interrupted by user")
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        self.display_statistics()
        
        if self.env:
            self.env.close()
            
        pygame.quit()
        print("\nThanks for playing! 🏎️")


def main():
    """Main function to run the manual racing game."""
    print("Initializing Manual CarRacing Game...")
    
    # Check if pygame is available
    try:
        import pygame
    except ImportError:
        print("❌ Pygame not found. Please install it with: pip install pygame")
        sys.exit(1)
        
    # Check if gym is available
    try:
        import gymnasium as gym
    except ImportError:
        print("❌ Gymnasium not found. Please install it with: pip install gymnasium[classic_control]")
        sys.exit(1)
        
    # Create and run the game
    game = ManualCarRacing(render_mode="human")
    game.run()


if __name__ == "__main__":
    main()