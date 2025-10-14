"""
A2C (Advantage Actor-Critic) Training Script for Super Mario Bros.

This script trains an A2C agent to play Super Mario Bros using a
synchronous advantage actor-critic algorithm.

Usage:
    python training/train_a2c.py --save-name mario_a2c.pth --epochs 10000 --lr 0.0001
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.src.env import create_train_env
from training.src.model import ActorCritic


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="A2C Training for Super Mario Bros"
    )

    # Environment settings
    parser.add_argument("--world", type=int, default=1,
                       help="World number (1-8)")
    parser.add_argument("--stage", type=int, default=1,
                       help="Stage number (1-4)")
    parser.add_argument("--action_type", type=str, default="complex",
                       choices=["right", "simple", "complex"],
                       help="Action space type")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor for rewards")
    parser.add_argument("--tau", type=float, default=1.0,
                       help="GAE parameter")
    parser.add_argument("--beta", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--num_local_steps", type=int, default=50,
                       help="Number of steps before update")
    parser.add_argument("--num-updates", type=int, default=10000,
                       help="Number of training updates (gradient updates)")

    # Saving and logging
    parser.add_argument("--save-name", type=str, default="mario_a2c.pth",
                       help="Name of the saved model file")
    parser.add_argument("--save-path", type=str, default="./models/saved_weights",
                       help="Directory to save model weights")
    parser.add_argument("--log-path", type=str, default="./logs/a2c_logs",
                       help="Directory to save training logs")
    parser.add_argument("--save-interval", type=int, default=100,
                       help="Save model every N updates")

    # Resume training
    parser.add_argument("--load", action="store_true",
                       help="Load a pretrained model to resume training")
    parser.add_argument("--model", type=str, default="",
                       help="Path to pretrained model file")

    args = parser.parse_args()
    return args


def train(args):
    """
    Main training loop for A2C.

    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(123)
    np.random.seed(123)

    # Create directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Determine device (cuda, mps, or cpu)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    print(f"Creating environment: World {args.world}-{args.stage}, Action type: {args.action_type}")
    env, num_states, num_actions = create_train_env(args.world, args.stage, args.action_type)

    # Create model
    model = ActorCritic(num_states, num_actions)
    model = model.to(device)

    # Load pretrained model if specified
    if args.load and args.model:
        if os.path.exists(args.model):
            print(f"Loading pretrained model from: {args.model}")
            if device == "cuda":
                model.load_state_dict(torch.load(args.model))
            else:
                model.load_state_dict(torch.load(args.model, map_location=device))
            print("✓ Model loaded successfully! Resuming training...")
        else:
            print(f"Error: Model file not found at {args.model}")
            sys.exit(1)

    model.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize state and LSTM hidden states
    state = torch.from_numpy(env.reset()).to(device)
    hx = torch.zeros(1, 512).to(device)
    cx = torch.zeros(1, 512).to(device)

    # Training statistics
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    best_reward = -float('inf')  # Track best average reward

    # Training loop
    print(f"\nStarting training for {args.num_updates} updates...")
    print("="*60)

    with tqdm(total=args.num_updates, desc="Training") as pbar:
        for update in range(args.num_updates):
            # Storage for trajectory
            log_policies = []
            values = []
            rewards = []
            entropies = []

            # Collect trajectory
            for step in range(args.num_local_steps):
                # Forward pass
                logits, value, hx, cx = model(state, hx, cx)
                policy = F.softmax(logits, dim=1)
                log_policy = F.log_softmax(logits, dim=1)
                entropy = -(policy * log_policy).sum(1, keepdim=True)

                # Sample action
                m = Categorical(policy)
                action = m.sample().item()

                # Take action in environment
                next_state, reward, done, info = env.step(action)
                next_state = torch.from_numpy(next_state).to(device)

                # Store trajectory
                values.append(value)
                log_policies.append(log_policy[0, action])
                rewards.append(reward)
                entropies.append(entropy)

                # Update statistics
                current_episode_reward += reward
                current_episode_length += 1

                # Handle episode end
                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)
                    current_episode_reward = 0
                    current_episode_length = 0

                    # Reset environment
                    state = torch.from_numpy(env.reset()).to(device)
                    hx = torch.zeros(1, 512).to(device)
                    cx = torch.zeros(1, 512).to(device)
                else:
                    state = next_state
                    hx = hx.detach()
                    cx = cx.detach()

                if done:
                    break

            # Compute returns and advantages
            R = torch.zeros(1, 1).to(device)
            if not done:
                _, R, _, _ = model(state, hx, cx)

            gae = torch.zeros(1, 1).to(device)
            actor_loss = 0
            critic_loss = 0
            entropy_loss = 0
            next_value = R

            # Backward pass through trajectory
            for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
                # GAE calculation
                gae = gae * args.gamma * args.tau
                gae = gae + reward + args.gamma * next_value.detach() - value.detach()
                next_value = value

                # Actor loss (policy gradient)
                actor_loss = actor_loss + log_policy * gae

                # Critic loss (value function)
                R = R * args.gamma + reward
                critic_loss = critic_loss + (R - value) ** 2 / 2

                # Entropy loss (exploration bonus)
                entropy_loss = entropy_loss + entropy

            # Total loss
            total_loss = -actor_loss + critic_loss - args.beta * entropy_loss

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Update progress bar
            pbar.update(1)
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                pbar.set_postfix({
                    'Episodes': len(episode_rewards),
                    'Avg Reward': f'{avg_reward:.2f}',
                    'Loss': f'{total_loss.item():.4f}'
                })

            # Save model periodically and check for best model
            if (update + 1) % args.save_interval == 0 and len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)

                # Save periodic checkpoint
                checkpoint_file = os.path.join(args.save_path, "mario_a3c_latest.pth")
                torch.save(model.state_dict(), checkpoint_file)
                tqdm.write(f"Update {update + 1}: Checkpoint saved to {checkpoint_file}")

                # Save best model if improved
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_file = os.path.join(args.save_path, "mario_a3c_best.pth")
                    torch.save(model.state_dict(), best_file)
                    tqdm.write(f"Update {update + 1}: New best model! Avg Reward: {avg_reward:.2f} (saved to {best_file})")

                # Save training statistics
                log_file = os.path.join(args.log_path, "training_log.txt")
                with open(log_file, "a") as f:
                    f.write(f"Update {update + 1}: Episodes={len(episode_rewards)}, "
                           f"Avg Reward={avg_reward:.2f}, Best Reward={best_reward:.2f}, "
                           f"Loss={total_loss.item():.4f}\n")

    # Final save
    final_file = os.path.join(args.save_path, "mario_a3c_final.pth")
    torch.save(model.state_dict(), final_file)

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Total episodes: {len(episode_rewards)}")
    if len(episode_rewards) > 0:
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        print(f"Average reward (last 100 episodes): {avg_reward:.2f}")
        print(f"Best average reward: {best_reward:.2f}")
    print(f"\nSaved models:")
    print(f"  - Latest: {os.path.join(args.save_path, 'mario_a3c_latest.pth')}")
    print(f"  - Best: {os.path.join(args.save_path, 'mario_a3c_best.pth')}")
    print(f"  - Final: {final_file}")
    print("="*60)

    env.close()


def main():
    """Main entry point."""
    args = get_args()

    # Validate arguments
    if not (1 <= args.world <= 8):
        print("Error: World must be between 1 and 8")
        sys.exit(1)
    if not (1 <= args.stage <= 4):
        print("Error: Stage must be between 1 and 4")
        sys.exit(1)

    print("\n" + "="*60)
    print("A2C TRAINING - SUPER MARIO BROS")
    print("="*60)
    print(f"World: {args.world}-{args.stage}")
    print(f"Action type: {args.action_type}")
    print(f"Learning rate: {args.lr}")
    print(f"Training updates: {args.num_updates}")
    print(f"Save path: {args.save_path}")
    print("="*60 + "\n")

    train(args)


if __name__ == "__main__":
    main()
