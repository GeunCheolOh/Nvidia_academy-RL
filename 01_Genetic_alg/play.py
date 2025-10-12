"""
Genetic Algorithm Snake Game Player with GUI
Usage: python play.py --model best_snake.npz --episodes 5
"""
import argparse
import os
import time
import numpy as np
import sys

from environments.snake_env import create_snake_environment
from algorithms.genetic import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description='Play Snake with trained Genetic Algorithm model')
    
    # Model settings
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    
    # Game settings
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to play')
    parser.add_argument('--board_width', type=int, default=15,
                       help='Snake game board width')
    parser.add_argument('--board_height', type=int, default=15,
                       help='Snake game board height')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--speed', type=float, default=0.1,
                       help='Game speed (delay between moves in seconds)')
    
    # Display settings
    parser.add_argument('--show_decisions', action='store_true',
                       help='Display neural network decisions')
    parser.add_argument('--show_stats', action='store_true',
                       help='Display episode statistics')
    parser.add_argument('--text_mode', action='store_true',
                       help='Use text-based display instead of GUI')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Genetic Algorithm Snake Player")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        return
    
    # Create environment with rendering
    try:
        env = create_snake_environment(
            width=args.board_width, 
            height=args.board_height, 
            render_mode=not args.text_mode
        )
        print(f"[OK] Snake environment created")
        print(f"   Board size: {args.board_width}x{args.board_height}")
        print(f"   Display mode: {'GUI' if not args.text_mode else 'Text'}")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return
    
    # Load trained model
    try:
        # Create neural network with appropriate structure
        input_size = env.get_simple_state_size()
        output_size = env.get_action_size()
        
        # Try to infer hidden sizes from model or use defaults
        network = NeuralNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=[128, 64]  # Default structure
        )
        
        network.load(args.model)
        print(f"[OK] Model loaded from {args.model}")
        print(f"   Input size: {input_size}")
        print(f"   Output size: {output_size}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Action names for display
    action_names = ["Up", "Right", "Down", "Left"]
    
    print(f"\nStarting {args.episodes} episodes...")
    if not args.text_mode:
        print("Close the game window or press Ctrl+C to stop early")
        print(f"Game speed: {args.speed}s per move")
    print()
    
    try:
        episode_scores = []
        episode_lengths = []
        episode_foods = []
        
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}")
            print("-" * 40)
            
            state = env.reset()
            total_reward = 0
            steps = 0
            food_collected = 0
            action_counts = {i: 0 for i in range(env.get_action_size())}
            
            episode_start_time = time.time()
            
            while steps < args.max_steps:
                # Get action from neural network
                simple_state = env.get_simple_state()
                action_probs = network.forward(simple_state)
                action = np.argmax(action_probs)
                action_counts[action] += 1
                
                # Show neural network decisions if requested
                if args.show_decisions:
                    print(f"Step {steps:3d}: {action_names[action]:<5} "
                          f"(confidence: {action_probs[action]:.3f})")
                    if steps % 10 == 0:  # Show full probabilities every 10 steps
                        prob_str = " | ".join([f"{name}: {prob:.3f}" 
                                             for name, prob in zip(action_names, action_probs)])
                        print(f"         Probabilities: {prob_str}")
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                total_reward += reward
                steps += 1
                current_score = info.get('score', 0)
                if current_score > food_collected:
                    food_collected = current_score
                
                # Render game
                if not args.text_mode:
                    env.render()
                    time.sleep(args.speed)
                else:
                    # Text mode: show board occasionally
                    if steps % 50 == 0 or done:
                        print(f"\nStep {steps}:")
                        print(env.get_board_string())
                
                state = next_state
                
                if done:
                    break
            
            episode_time = time.time() - episode_start_time
            episode_scores.append(current_score)
            episode_lengths.append(steps)
            episode_foods.append(food_collected)
            
            # Episode summary
            print(f"\nEpisode {episode + 1} completed:")
            print(f"  Final Score: {current_score}")
            print(f"  Food Collected: {food_collected}")
            print(f"  Steps: {steps}")
            print(f"  Time: {episode_time:.1f}s")
            print(f"  Total Reward: {total_reward:.1f}")
            
            if args.show_stats:
                print(f"  Action Distribution:")
                for i, count in action_counts.items():
                    if count > 0:
                        percentage = (count / steps) * 100
                        print(f"    {action_names[i]:<5}: {count:3d} ({percentage:5.1f}%)")
                
                # Efficiency metrics
                if steps > 0:
                    efficiency = food_collected / steps * 100
                    print(f"  Efficiency: {efficiency:.2f}% (food/step)")
            
            print()
        
        # Overall summary
        print("=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"Episodes played: {len(episode_scores)}")
        print(f"Average score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
        print(f"Best score: {max(episode_scores)}")
        print(f"Average food collected: {np.mean(episode_foods):.2f}")
        print(f"Average game length: {np.mean(episode_lengths):.1f} steps")
        
        # Success metrics
        good_games = sum(1 for score in episode_scores if score >= 5)
        great_games = sum(1 for score in episode_scores if score >= 10)
        print(f"Good games (≥5 food): {good_games}/{len(episode_scores)} ({good_games/len(episode_scores):.1%})")
        print(f"Great games (≥10 food): {great_games}/{len(episode_scores)} ({great_games/len(episode_scores):.1%})")
        
        # Efficiency analysis
        if episode_lengths:
            total_efficiency = sum(episode_foods) / sum(episode_lengths) * 100
            print(f"Overall efficiency: {total_efficiency:.2f}% (food/step)")
    
    except KeyboardInterrupt:
        print("\n[WARNING] Game interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] Game failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\nGame session ended")


if __name__ == "__main__":
    main()

