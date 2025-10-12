"""
Genetic Algorithm Training Script for Snake Game
Usage: python train.py --generations 100 --population_size 50 --save_model best_snake.npz
"""
import argparse
import os
import time
import numpy as np
import sys

from environments.snake_env import create_snake_environment
from algorithms.genetic import GeneticAlgorithm
from core.base import Trainer
from utils.visualization import EvolutionVisualizer, StatisticsCollector, save_evolution_config


def main():
    parser = argparse.ArgumentParser(description='Train Genetic Algorithm on Snake Game')
    
    # Algorithm settings
    parser.add_argument('--generations', type=int, default=100,
                       help='Number of generations to evolve')
    parser.add_argument('--population_size', type=int, default=50,
                       help='Population size')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                       help='Mutation rate')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                       help='Crossover rate')
    parser.add_argument('--elite_size', type=int, default=5,
                       help='Number of elite individuals to preserve')
    
    # Neural network settings
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer sizes')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh', 'sigmoid'],
                       help='Activation function')
    
    # Environment settings
    parser.add_argument('--board_width', type=int, default=15,
                       help='Snake game board width')
    parser.add_argument('--board_height', type=int, default=15,
                       help='Snake game board height')
    parser.add_argument('--episodes_per_individual', type=int, default=3,
                       help='Number of episodes to evaluate each individual')
    
    # Selection methods
    parser.add_argument('--selection_method', type=str, default='tournament',
                       choices=['roulette', 'tournament', 'rank'],
                       help='Selection method')
    parser.add_argument('--tournament_size', type=int, default=5,
                       help='Tournament size for tournament selection')
    
    # Saving/Loading
    parser.add_argument('--save_model', type=str, default='models/snake_ga.npz',
                       help='Path to save best model')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save model every N generations')
    parser.add_argument('--log_interval', type=int, default=5,
                       help='Print progress every N generations')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Create evolution plots')
    parser.add_argument('--results_dir', type=str, default='results/',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Genetic Algorithm Training on Snake Game")
    print("=" * 60)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    
    # Create environment
    try:
        env = create_snake_environment(
            width=args.board_width, 
            height=args.board_height, 
            render_mode=False
        )
        print(f"[OK] Snake environment created")
        print(f"   Board size: {args.board_width}x{args.board_height}")
        print(f"   State size: {env.get_simple_state_size()}")
        print(f"   Action size: {env.get_action_size()}")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return
    
    # Create genetic algorithm
    ga = GeneticAlgorithm(
        population_size=args.population_size,
        input_size=env.get_simple_state_size(),
        output_size=env.get_action_size(),
        hidden_sizes=args.hidden_layers,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        selection_method=args.selection_method,
        tournament_size=args.tournament_size,
        elite_size=args.elite_size
    )
    
    print(f"\nGenetic Algorithm Configuration:")
    print(f"   Population size: {args.population_size}")
    print(f"   Mutation rate: {args.mutation_rate}")
    print(f"   Crossover rate: {args.crossover_rate}")
    print(f"   Selection method: {args.selection_method}")
    print(f"   Elite preservation: {args.elite_size}")
    print(f"   Neural network: {env.get_simple_state_size()} -> {args.hidden_layers} -> {env.get_action_size()}")
    
    # Save training configuration
    config = {
        'generations': args.generations,
        'population_size': args.population_size,
        'mutation_rate': args.mutation_rate,
        'crossover_rate': args.crossover_rate,
        'elite_size': args.elite_size,
        'hidden_layers': args.hidden_layers,
        'activation': args.activation,
        'board_size': f"{args.board_width}x{args.board_height}",
        'episodes_per_individual': args.episodes_per_individual,
        'selection_method': args.selection_method,
        'tournament_size': args.tournament_size,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    save_evolution_config(config, os.path.join(args.results_dir, 'config.txt'))
    
    # Create trainer and tools
    trainer = Trainer(ga, env)
    visualizer = EvolutionVisualizer(save_dir=args.results_dir)
    stats = StatisticsCollector()
    
    print(f"\nStarting evolution for {args.generations} generations...")
    start_time = time.time()
    
    try:
        best_fitness = float('-inf')
        best_individual = None
        
        for generation in range(args.generations):
            # Evaluate population
            fitness_scores = []
            performance_metrics = {
                'scores': [],
                'lengths': [],
                'survival_count': 0
            }
            
            for individual in ga.population:
                # Evaluate individual over multiple episodes
                episode_scores = []
                episode_lengths = []
                
                for episode in range(args.episodes_per_individual):
                    state = env.reset()
                    total_reward = 0
                    steps = 0
                    game_score = 0
                    
                    while steps < 500:  # Max steps per episode
                        # Get action from neural network
                        simple_state = env.get_simple_state()
                        action_probs = individual.forward(simple_state)
                        action = np.argmax(action_probs)
                        
                        state, reward, done, info = env.step(action)
                        total_reward += reward
                        steps += 1
                        game_score = info.get('score', 0)
                        
                        if done:
                            break
                    
                    episode_scores.append(game_score)
                    episode_lengths.append(steps)
                
                # Calculate fitness (average performance)
                avg_score = np.mean(episode_scores)
                avg_length = np.mean(episode_lengths)
                
                # Fitness function: prioritize food collection with length bonus
                fitness = avg_score * 100 + avg_length * 0.1
                fitness_scores.append(fitness)
                
                # Performance tracking
                performance_metrics['scores'].append(avg_score)
                performance_metrics['lengths'].append(avg_length)
                if avg_score > 0:
                    performance_metrics['survival_count'] += 1
            
            # Track best individual
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                best_individual = ga.population[best_idx]
                
                # Save best model
                best_model_path = args.save_model.replace('.npz', '_best.npz')
                best_individual.save(best_model_path)
            
            # Record statistics
            avg_score = np.mean(performance_metrics['scores'])
            max_score = max(performance_metrics['scores'])
            avg_length = np.mean(performance_metrics['lengths'])
            max_length = max(performance_metrics['lengths'])
            survival_rate = performance_metrics['survival_count'] / len(ga.population)
            
            stats.record_generation(
                fitness_scores=fitness_scores,
                diversity_score=np.std(fitness_scores),
                performance_metrics={
                    'avg_score': avg_score,
                    'max_score': max_score,
                    'avg_length': avg_length,
                    'max_length': max_length,
                    'survival_rate': survival_rate
                }
            )
            
            # Logging
            if (generation + 1) % args.log_interval == 0:
                print(f"Generation {generation + 1:3d}/{args.generations} | "
                      f"Best Fitness: {max_fitness:8.2f} | "
                      f"Avg Score: {avg_score:6.2f} | "
                      f"Max Score: {max_score:3.0f} | "
                      f"Survival: {survival_rate:.1%}")
            
            # Save model periodically
            if (generation + 1) % args.save_interval == 0:
                if best_individual:
                    best_individual.save(args.save_model)
                print(f"Model saved at generation {generation + 1}")
            
            # Evolve population (except last generation)
            if generation < args.generations - 1:
                ga.evolve(fitness_scores)
        
        training_time = time.time() - start_time
        
        # Final save
        if best_individual:
            best_individual.save(args.save_model)
        
        print(f"\n[OK] Evolution completed in {training_time:.1f}s")
        print(f"   Best fitness: {best_fitness:.2f}")
        print(f"   Final 10-generation average: {np.mean([stats.generation_stats[i]['avg_fitness'] for i in range(-10, 0)]):.2f}")
        
        # Statistics summary
        stats.print_summary()
        
        # Visualization
        if args.visualize:
            print("\nCreating evolution plots...")
            
            # Evolution progress
            visualizer.plot_evolution_progress(
                generation_stats=stats.generation_stats,
                title="Snake Genetic Algorithm Evolution",
                save_name="evolution_progress"
            )
            
            # Performance metrics
            visualizer.plot_snake_performance(
                performance_data=stats.performance_history,
                title="Snake Performance Over Generations",
                save_name="snake_performance"
            )
            
            # Final generation fitness distribution
            if stats.generation_stats:
                final_fitness = [individual.fitness for individual in ga.population if hasattr(individual, 'fitness')]
                if not final_fitness:
                    final_fitness = fitness_scores
                
                visualizer.plot_fitness_distribution(
                    fitness_scores=final_fitness,
                    generation=args.generations,
                    title="Final Generation Fitness Distribution",
                    save_name="final_fitness_distribution"
                )
            
            print(f"Plots saved in {args.results_dir}")
    
    except KeyboardInterrupt:
        print(f"\n[WARNING] Evolution interrupted by user")
        # Save partial model
        if best_individual:
            interrupted_path = args.save_model.replace('.npz', '_interrupted.npz')
            best_individual.save(interrupted_path)
            print(f"Partial model saved to {interrupted_path}")
    
    except Exception as e:
        print(f"\n[ERROR] Evolution failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\nEvolution session ended")


if __name__ == "__main__":
    main()

