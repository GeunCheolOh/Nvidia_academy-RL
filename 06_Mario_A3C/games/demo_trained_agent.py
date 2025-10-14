"""
Demo script for trained A2C agent playing Super Mario Bros.

This script loads a trained model and demonstrates its performance
by playing the game. You can compare the trained agent with a random agent.

Usage:
    # Demo trained agent
    python games/demo_trained_agent.py --model-path models/saved_weights/mario_a2c.pth

    # Compare with random agent
    python games/demo_trained_agent.py --model-path models/saved_weights/mario_a2c.pth --compare
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.src.env import create_train_env
from training.src.model import ActorCritic


def play_episode(env, model, device, render=True, max_steps=5000):
    """
    Play one episode with the given model.

    Args:
        env: Game environment
        model: Trained model (None for random agent)
        device: Device to run model on
        render: Whether to render the game
        max_steps: Maximum steps per episode

    Returns:
        Tuple of (total_reward, steps, info)
    """
    state = torch.from_numpy(env.reset()).to(device)
    done = False
    total_reward = 0
    steps = 0

    # Initialize LSTM hidden states
    if model is not None:
        hx = torch.zeros(1, 512).to(device)
        cx = torch.zeros(1, 512).to(device)

    while not done and steps < max_steps:
        if render:
            env.render()
            time.sleep(0.016)  # ~60 FPS

        if model is not None:
            # Use trained model
            with torch.no_grad():
                logits, value, hx, cx = model(state, hx, cx)
                policy = F.softmax(logits, dim=1)
                action = torch.argmax(policy).item()
        else:
            # Random action
            action = env.action_space.sample()

        # Take action
        next_state, reward, done, info = env.step(action)
        state = torch.from_numpy(next_state).to(device)
        total_reward += reward
        steps += 1

        # Check if level completed
        if info.get("flag_get", False):
            print(f"\n{'='*60}")
            print(f"🎉 LEVEL COMPLETED! 🎉")
            print(f"{'='*60}")
            break

    return total_reward, steps, info


def demo_agent(args):
    """
    Demonstrate the trained agent.

    Args:
        args: Command line arguments
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    print(f"Creating environment: World {args.world}-{args.stage}")
    env, num_states, num_actions = create_train_env(args.world, args.stage, args.action_type)

    # Load trained model
    print(f"Loading model from: {args.model_path}")
    model = ActorCritic(num_states, num_actions)

    if os.path.exists(args.model_path):
        if device == "cuda":
            model.load_state_dict(torch.load(args.model_path))
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully!")
    else:
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    print("\n" + "="*60)
    print("TRAINED AGENT DEMO")
    print("="*60)
    print(f"Playing {args.num_episodes} episode(s)...")
    print("="*60 + "\n")

    # Play episodes with trained agent
    total_rewards = []
    total_steps = []
    total_scores = []
    flags_captured = 0

    for episode in range(args.num_episodes):
        print(f"\nEpisode {episode + 1}/{args.num_episodes}")
        print("-" * 40)

        reward, steps, info = play_episode(env, model, device, render=args.render, max_steps=args.max_steps)

        total_rewards.append(reward)
        total_steps.append(steps)
        total_scores.append(info['score'])

        if info.get("flag_get", False):
            flags_captured += 1

        print(f"Steps: {steps}")
        print(f"Score: {info['score']}")
        print(f"Reward: {reward:.2f}")
        print(f"Max X Position: {info['x_pos']}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY - TRAINED AGENT")
    print("="*60)
    print(f"Episodes played: {args.num_episodes}")
    print(f"Levels completed: {flags_captured}/{args.num_episodes}")
    print(f"Average reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Average score: {sum(total_scores)/len(total_scores):.2f}")
    print(f"Average steps: {sum(total_steps)/len(total_steps):.2f}")
    print("="*60)

    # Compare with random agent if requested
    if args.compare:
        print("\n" + "="*60)
        print("RANDOM AGENT COMPARISON")
        print("="*60)
        print(f"Playing {args.num_episodes} episode(s) with random actions...")
        print("="*60 + "\n")

        random_rewards = []
        random_steps = []
        random_scores = []
        random_flags = 0

        for episode in range(args.num_episodes):
            print(f"\nRandom Episode {episode + 1}/{args.num_episodes}")
            print("-" * 40)

            reward, steps, info = play_episode(env, None, device, render=args.render, max_steps=args.max_steps)

            random_rewards.append(reward)
            random_steps.append(steps)
            random_scores.append(info['score'])

            if info.get("flag_get", False):
                random_flags += 1

            print(f"Steps: {steps}")
            print(f"Score: {info['score']}")
            print(f"Reward: {reward:.2f}")
            print(f"Max X Position: {info['x_pos']}")

        # Comparison summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Metric':<20} {'Trained Agent':<20} {'Random Agent':<20}")
        print("-" * 60)
        print(f"{'Levels completed':<20} {flags_captured}/{args.num_episodes:<20} {random_flags}/{args.num_episodes:<20}")
        print(f"{'Avg Reward':<20} {sum(total_rewards)/len(total_rewards):<20.2f} {sum(random_rewards)/len(random_rewards):<20.2f}")
        print(f"{'Avg Score':<20} {sum(total_scores)/len(total_scores):<20.2f} {sum(random_scores)/len(random_scores):<20.2f}")
        print(f"{'Avg Steps':<20} {sum(total_steps)/len(total_steps):<20.2f} {sum(random_steps)/len(random_steps):<20.2f}")
        print("="*60)

    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Demo trained A2C agent")

    # Model settings
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model file")

    # Environment settings
    parser.add_argument("--world", type=int, default=1,
                       help="World number (1-8)")
    parser.add_argument("--stage", type=int, default=1,
                       help="Stage number (1-4)")
    parser.add_argument("--action_type", type=str, default="complex",
                       choices=["right", "simple", "complex"],
                       help="Action space type")

    # Demo settings
    parser.add_argument("--num-episodes", type=int, default=5,
                       help="Number of episodes to play")
    parser.add_argument("--max-steps", type=int, default=5000,
                       help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", default=True,
                       help="Render the game")
    parser.add_argument("--no-render", action="store_false", dest="render",
                       help="Don't render the game")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with random agent")

    args = parser.parse_args()

    # Validate arguments
    if not (1 <= args.world <= 8):
        print("Error: World must be between 1 and 8")
        sys.exit(1)
    if not (1 <= args.stage <= 4):
        print("Error: Stage must be between 1 and 4")
        sys.exit(1)

    print("\n" + "="*60)
    print("A2C AGENT DEMO - SUPER MARIO BROS")
    print("="*60)
    print(f"World: {args.world}-{args.stage}")
    print(f"Action type: {args.action_type}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Render: {args.render}")
    print(f"Compare mode: {args.compare}")
    print("="*60 + "\n")

    demo_agent(args)


if __name__ == "__main__":
    main()
