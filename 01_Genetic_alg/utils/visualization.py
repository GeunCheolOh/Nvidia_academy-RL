"""
Visualization and Statistics Tools for Genetic Algorithm Training
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import os


class EvolutionVisualizer:
    """Evolution Visualization Class"""
    
    def __init__(self, save_dir: str = "results/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set matplotlib parameters for English
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('default')
    
    def plot_evolution_progress(self, generation_stats: List[Dict],
                              title: str = "Genetic Algorithm Evolution",
                              save_name: Optional[str] = None):
        """Plot evolution progress over generations"""
        
        generations = list(range(1, len(generation_stats) + 1))
        max_fitness = [stats['max_fitness'] for stats in generation_stats]
        avg_fitness = [stats['avg_fitness'] for stats in generation_stats]
        min_fitness = [stats['min_fitness'] for stats in generation_stats]
        
        plt.figure(figsize=(12, 8))
        
        # Plot fitness curves
        plt.plot(generations, max_fitness, 'g-', linewidth=2, label='Best Fitness')
        plt.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average Fitness')
        plt.plot(generations, min_fitness, 'r-', linewidth=1, alpha=0.7, label='Worst Fitness')
        
        # Fill between for variance visualization
        plt.fill_between(generations, min_fitness, max_fitness, alpha=0.2, color='blue')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        if generation_stats:
            final_stats = generation_stats[-1]
            stats_text = f"Final Generation:\nBest: {final_stats['max_fitness']:.1f}\nAvg: {final_stats['avg_fitness']:.1f}"
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_fitness_distribution(self, fitness_scores: List[float], generation: int,
                                title: str = "Fitness Distribution",
                                save_name: Optional[str] = None):
        """Plot fitness distribution for a generation"""
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(fitness_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(fitness_scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(fitness_scores):.2f}')
        plt.axvline(np.max(fitness_scores), color='green', linestyle='--', 
                   linewidth=2, label=f'Best: {np.max(fitness_scores):.2f}')
        
        plt.xlabel('Fitness Score')
        plt.ylabel('Number of Individuals')
        plt.title(f'{title} - Generation {generation}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Population: {len(fitness_scores)}\nStd: {np.std(fitness_scores):.2f}"
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}_gen{generation}.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_diversity_metrics(self, diversity_data: List[float],
                             title: str = "Population Diversity",
                             save_name: Optional[str] = None):
        """Plot population diversity over generations"""
        
        generations = list(range(1, len(diversity_data) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, diversity_data, 'purple', linewidth=2, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Diversity Score')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(diversity_data) > 1:
            z = np.polyfit(generations, diversity_data, 1)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), "r--", alpha=0.8, 
                    label=f'Trend (slope: {z[0]:.4f})')
            plt.legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_selection_pressure(self, selection_data: Dict[str, List[float]],
                              title: str = "Selection Pressure Analysis",
                              save_name: Optional[str] = None):
        """Plot selection pressure metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title)
        
        # Selection rates
        if 'selection_rates' in selection_data:
            axes[0, 0].plot(selection_data['selection_rates'], 'b-', linewidth=2)
            axes[0, 0].set_title('Selection Rate')
            axes[0, 0].set_xlabel('Generation')
            axes[0, 0].set_ylabel('Rate')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Mutation rates
        if 'mutation_rates' in selection_data:
            axes[0, 1].plot(selection_data['mutation_rates'], 'r-', linewidth=2)
            axes[0, 1].set_title('Mutation Rate')
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Rate')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Crossover success
        if 'crossover_success' in selection_data:
            axes[1, 0].plot(selection_data['crossover_success'], 'g-', linewidth=2)
            axes[1, 0].set_title('Crossover Success Rate')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Elite preservation
        if 'elite_fitness' in selection_data:
            axes[1, 1].plot(selection_data['elite_fitness'], 'm-', linewidth=2)
            axes[1, 1].set_title('Elite Fitness')
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Fitness')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_snake_performance(self, performance_data: Dict[str, List],
                             title: str = "Snake Performance Metrics",
                             save_name: Optional[str] = None):
        """Plot Snake-specific performance metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title)
        
        generations = list(range(1, len(performance_data['avg_score']) + 1))
        
        # Average scores
        axes[0, 0].plot(generations, performance_data['avg_score'], 'b-', linewidth=2, label='Average')
        axes[0, 0].plot(generations, performance_data['max_score'], 'g-', linewidth=2, label='Best')
        axes[0, 0].set_title('Snake Scores')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Game lengths
        axes[0, 1].plot(generations, performance_data['avg_length'], 'r-', linewidth=2, label='Average')
        axes[0, 1].plot(generations, performance_data['max_length'], 'orange', linewidth=2, label='Best')
        axes[0, 1].set_title('Game Length (Steps)')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Survival rates
        if 'survival_rate' in performance_data:
            axes[1, 0].plot(generations, performance_data['survival_rate'], 'purple', linewidth=2)
            axes[1, 0].set_title('Survival Rate')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Food collection efficiency
        if 'food_efficiency' in performance_data:
            axes[1, 1].plot(generations, performance_data['food_efficiency'], 'brown', linewidth=2)
            axes[1, 1].set_title('Food Collection Efficiency')
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Efficiency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()


class StatisticsCollector:
    """Statistics Collection Class for Genetic Algorithm"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset statistics"""
        self.generation_stats = []
        self.diversity_history = []
        self.selection_history = {}
        self.performance_history = {
            'avg_score': [],
            'max_score': [],
            'avg_length': [],
            'max_length': [],
            'survival_rate': [],
            'food_efficiency': []
        }
    
    def record_generation(self, fitness_scores: List[float], 
                         diversity_score: float = None,
                         selection_metrics: Dict = None,
                         performance_metrics: Dict = None):
        """Record generation statistics"""
        
        # Basic fitness statistics
        gen_stats = {
            'generation': len(self.generation_stats) + 1,
            'max_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'min_fitness': min(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'population_size': len(fitness_scores)
        }
        
        self.generation_stats.append(gen_stats)
        
        # Diversity tracking
        if diversity_score is not None:
            self.diversity_history.append(diversity_score)
        
        # Selection metrics
        if selection_metrics:
            for key, value in selection_metrics.items():
                if key not in self.selection_history:
                    self.selection_history[key] = []
                self.selection_history[key].append(value)
        
        # Performance metrics
        if performance_metrics:
            for key, value in performance_metrics.items():
                if key in self.performance_history:
                    self.performance_history[key].append(value)
    
    def get_summary(self, last_n_generations: int = 10) -> Dict:
        """Get statistics summary"""
        if not self.generation_stats:
            return {}
        
        recent_stats = self.generation_stats[-last_n_generations:]
        
        summary = {
            'total_generations': len(self.generation_stats),
            'best_fitness_ever': max(stat['max_fitness'] for stat in self.generation_stats),
            'recent_avg_fitness': np.mean([stat['avg_fitness'] for stat in recent_stats]),
            'recent_max_fitness': np.mean([stat['max_fitness'] for stat in recent_stats]),
            'fitness_improvement': 0,
            'population_size': self.generation_stats[-1]['population_size']
        }
        
        # Calculate improvement
        if len(self.generation_stats) >= 2:
            early_avg = np.mean([stat['avg_fitness'] for stat in self.generation_stats[:5]])
            late_avg = np.mean([stat['avg_fitness'] for stat in self.generation_stats[-5:]])
            summary['fitness_improvement'] = late_avg - early_avg
        
        return summary
    
    def print_summary(self, last_n_generations: int = 10):
        """Print statistics summary"""
        summary = self.get_summary(last_n_generations)
        
        if not summary:
            print("No statistics collected yet.")
            return
        
        print(f"Evolution Statistics (Last {last_n_generations} generations)")
        print(f"   Total Generations: {summary['total_generations']}")
        print(f"   Best Fitness Ever: {summary['best_fitness_ever']:.2f}")
        print(f"   Recent Average Fitness: {summary['recent_avg_fitness']:.2f}")
        print(f"   Recent Best Fitness: {summary['recent_max_fitness']:.2f}")
        print(f"   Fitness Improvement: {summary['fitness_improvement']:+.2f}")
        print(f"   Population Size: {summary['population_size']}")
    
    def print_generation_summary(self, generation: int):
        """Print summary for specific generation"""
        if generation <= 0 or generation > len(self.generation_stats):
            print(f"Generation {generation} not found.")
            return
        
        stats = self.generation_stats[generation - 1]
        print(f"Generation {generation}:")
        print(f"  Best Fitness: {stats['max_fitness']:.2f}")
        print(f"  Average Fitness: {stats['avg_fitness']:.2f}")
        print(f"  Worst Fitness: {stats['min_fitness']:.2f}")
        print(f"  Standard Deviation: {stats['std_fitness']:.2f}")


def save_evolution_config(config: Dict, save_path: str = "results/config.txt"):
    """Save evolution configuration"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Snake Genetic Algorithm Configuration\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 50 + "\n")


def benchmark_population(population, env, num_episodes: int = 3) -> Dict:
    """Benchmark population performance"""
    import time
    
    print(f"Population Benchmark ({num_episodes} episodes per individual)")
    
    total_scores = []
    total_time = 0
    
    for i, individual in enumerate(population[:5]):  # Test first 5 individuals
        episode_scores = []
        
        start_time = time.time()
        for episode in range(num_episodes):
            state = env.reset()
            score = 0
            steps = 0
            
            while steps < 1000:  # Max steps per episode
                # Get action from individual (assuming neural network)
                if hasattr(individual, 'forward'):
                    action_probs = individual.forward(env.get_simple_state())
                    action = np.argmax(action_probs)
                else:
                    action = np.random.randint(env.get_action_size())
                
                state, reward, done, info = env.step(action)
                score += reward
                steps += 1
                
                if done:
                    break
            
            episode_scores.append(info.get('score', 0))
        
        individual_time = time.time() - start_time
        avg_score = np.mean(episode_scores)
        total_scores.extend(episode_scores)
        total_time += individual_time
        
        print(f"  Individual {i + 1}: {avg_score:.1f} avg score, {individual_time:.2f}s")
    
    benchmark_results = {
        "avg_score": np.mean(total_scores),
        "std_score": np.std(total_scores),
        "total_time": total_time,
        "episodes_tested": len(total_scores)
    }
    
    print(f"\nBenchmark Results:")
    print(f"   Average Score: {benchmark_results['avg_score']:.2f} ± {benchmark_results['std_score']:.2f}")
    print(f"   Total Time: {benchmark_results['total_time']:.2f}s")
    print(f"   Episodes Tested: {benchmark_results['episodes_tested']}")
    
    return benchmark_results