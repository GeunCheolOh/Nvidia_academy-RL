"""
Train - 유전 알고리즘으로 뱀게임 학습 (로그 및 플롯 기능 추가)
"""
import pygame
import random
import numpy as np
from copy import deepcopy
from snake_game import SnakeGame, SCREEN_SIZE, PIXEL_SIZE
from genome import Genome
import argparse
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime


def plot_training_progress(log_file, save_path='logs/training_plot.png'):
    """학습 진행 상황을 플롯으로 저장"""
    # 로그 파일 읽기
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    generations = [log['generation'] for log in logs]
    best_fitness = [log['best_fitness'] for log in logs]
    avg_fitness = [log['avg_fitness'] for log in logs]
    avg_score = [log['avg_score'] for log in logs]
    
    # 플롯 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Fitness 그래프
    ax1.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness')
    ax1.plot(generations, avg_fitness, 'b-', linewidth=2, label='Avg Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Score 그래프
    ax2.plot(generations, avg_score, 'r-', linewidth=2, label='Avg Score')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Score')
    ax2.set_title('Score Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [Plot saved] {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Snake AI with Genetic Algorithm')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--n_best', type=int, default=5, help='Number of best genomes to keep')
    parser.add_argument('--n_children', type=int, default=5, help='Number of children to generate')
    parser.add_argument('--mutation_prob', type=float, default=0.4, help='Mutation probability')
    parser.add_argument('--generations', type=int, default=1000, help='Number of generations')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N generations')
    parser.add_argument('--log_interval', type=int, default=1, help='Log every N generations')
    parser.add_argument('--plot_interval', type=int, default=10, help='Generate plot every N generations')
    parser.add_argument('--render', action='store_true', help='Enable visualization during training (slower)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of genomes to evaluate per batch (useful when render is off)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Genetic Algorithm Training - Snake Game")
    print("=" * 60)
    print(f"Population: {args.population}")
    print(f"Best genomes: {args.n_best}")
    print(f"Children per generation: {args.n_children}")
    print(f"Mutation probability: {args.mutation_prob}")
    print(f"Max generations: {args.generations}")
    print(f"Save interval: {args.save_interval}")
    print(f"Plot interval: {args.plot_interval}")
    print(f"Render mode: {'ON (slower, visual)' if args.render else 'OFF (faster)'}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 로그 파일 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_log_{timestamp}.json'
    training_logs = []
    
    # Pygame 초기화 (렌더링 모드에서만)
    screen = None
    if args.render:
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((SCREEN_SIZE * PIXEL_SIZE, 
                                         SCREEN_SIZE * PIXEL_SIZE))
        pygame.display.set_caption('Snake - Training')
    else:
        # 렌더링 없이 학습할 때는 dummy screen
        pygame.init()  # pygame은 초기화 필요
        screen = pygame.Surface((SCREEN_SIZE * PIXEL_SIZE, SCREEN_SIZE * PIXEL_SIZE))
    
    # 초기 개체군 생성
    genomes = [Genome() for _ in range(args.population)]
    best_genomes = None
    best_fitness_ever = 0
    best_score_ever = 0
    
    n_gen = 0
    try:
        while n_gen < args.generations:
            n_gen += 1
            
            # 각 개체 평가
            for i, genome in enumerate(genomes):
                game = SnakeGame(screen, genome=genome)
                fitness, score = game.run(manual=False, training_mode=True, render=args.render)
                
                # TODO 1: 적합도 평가 (Fitness Evaluation)
                # 힌트: 게임에서 반환된 fitness 값을 genome 객체에 저장하세요
                # genome.fitness = ?
                # YOUR CODE HERE
                raise NotImplementedError("TODO 1: 적합도를 genome에 저장하세요")
                
                # 최고 기록 갱신
                if fitness > best_fitness_ever:
                    best_fitness_ever = fitness
                    best_score_ever = score
                
                # 배치 단위로 진행 상황 표시 (렌더링 off일 때만)
                if not args.render and (i + 1) % args.batch_size == 0:
                    print(f"  Gen {n_gen} - Evaluating: {i + 1}/{len(genomes)} genomes", end='\r')
            
            # 진행 상황 표시 후 줄바꿈
            if not args.render:
                print()  # 줄바꿈
            
            # 기존 최고 유전자 추가
            if best_genomes is not None:
                genomes.extend(best_genomes)
            
            # TODO 2: 선택 (Selection) - Fitness 기준 정렬
            # 힌트: genomes 리스트를 fitness가 높은 순서대로 정렬하세요
            # genomes.sort(key=lambda x: x.?, reverse=?)
            # YOUR CODE HERE
            raise NotImplementedError("TODO 2: genomes를 fitness 기준으로 내림차순 정렬하세요")
            
            # 통계 계산
            avg_fitness = np.mean([g.fitness for g in genomes[:args.population]])
            avg_score = np.mean([g.fitness / 10 for g in genomes[:args.population] if g.fitness > 0])
            max_fitness = genomes[0].fitness
            
            # 로그 기록
            if n_gen % args.log_interval == 0:
                log_entry = {
                    'generation': n_gen,
                    'best_fitness': float(max_fitness),
                    'avg_fitness': float(avg_fitness),
                    'avg_score': float(avg_score),
                    'best_fitness_ever': float(best_fitness_ever),
                    'best_score_ever': int(best_score_ever),
                    'timestamp': datetime.now().isoformat()
                }
                training_logs.append(log_entry)
                
                # 로그 파일 저장
                with open(log_file, 'w') as f:
                    json.dump(training_logs, f, indent=2)
            
            # 진행 상황 출력
            print(f"Gen {n_gen:4d} | Best: {max_fitness:7.1f} | "
                  f"Avg: {avg_fitness:6.1f} | Score: {avg_score:4.1f} | "
                  f"Best Ever: {best_fitness_ever:7.1f} (Score: {best_score_ever})")
            
            # 최고 유전자 저장
            best_genomes = deepcopy(genomes[:args.n_best])
            
            # 주기적으로 모델 저장
            if n_gen % args.save_interval == 0:
                model_path = f'models/snake_ga_gen{n_gen}.npz'
                best_genomes[0].save(model_path)
                print(f"  [Saved] {model_path}")
            
            # 주기적으로 플롯 생성
            if n_gen % args.plot_interval == 0 and len(training_logs) > 1:
                try:
                    plot_training_progress(log_file, f'logs/training_plot_gen{n_gen}.png')
                except Exception as e:
                    print(f"  [Warning] Plot generation failed: {e}")
            
            # TODO 3: 교차 (Crossover)
            # 힌트: 두 부모 genome의 가중치를 조합하여 자식을 만듭니다
            # 단일점 교차(Single-point crossover)를 사용하세요
            for i in range(args.n_children):
                new_genome = deepcopy(best_genomes[0])
                a_genome = random.choice(best_genomes)
                b_genome = random.choice(best_genomes)
                
                # w1 교차 예시 (나머지 w2, w3, w4도 같은 방식으로 구현)
                cut = random.randint(0, new_genome.w1.shape[1])
                # YOUR CODE HERE
                # new_genome.w1[:, :cut] = ?
                # new_genome.w1[:, cut:] = ?
                raise NotImplementedError("TODO 3-1: w1 교차를 구현하세요")
                
                # w2 교차
                cut = random.randint(0, new_genome.w2.shape[1])
                # YOUR CODE HERE
                raise NotImplementedError("TODO 3-2: w2 교차를 구현하세요")
                
                # w3 교차
                cut = random.randint(0, new_genome.w3.shape[1])
                # YOUR CODE HERE
                raise NotImplementedError("TODO 3-3: w3 교차를 구현하세요")
                
                # w4 교차
                cut = random.randint(0, new_genome.w4.shape[1])
                # YOUR CODE HERE
                raise NotImplementedError("TODO 3-4: w4 교차를 구현하세요")
                
                best_genomes.append(new_genome)
            
            # TODO 4: 돌연변이 (Mutation)
            # 힌트: 가중치에 랜덤 노이즈를 추가하여 새로운 변이를 생성합니다
            genomes = []
            for i in range(int(args.population / (args.n_best + args.n_children))):
                for bg in best_genomes:
                    new_genome = deepcopy(bg)
                    
                    mean = 20
                    stddev = 10
                    
                    # w1 돌연변이 예시 (나머지 w2, w3, w4도 같은 방식으로 구현)
                    if random.uniform(0, 1) < args.mutation_prob:
                        # YOUR CODE HERE
                        # new_genome.w1 += new_genome.w1 * np.random.normal(...) / 100 * np.random.randint(-1, 2, ...)
                        raise NotImplementedError("TODO 4-1: w1 돌연변이를 구현하세요")
                    
                    if random.uniform(0, 1) < args.mutation_prob:
                        # YOUR CODE HERE
                        # 힌트: w2의 shape은 (10, 20)입니다
                        raise NotImplementedError("TODO 4-2: w2 돌연변이를 구현하세요")
                    
                    if random.uniform(0, 1) < args.mutation_prob:
                        # YOUR CODE HERE
                        # 힌트: w3의 shape은 (20, 10)입니다
                        raise NotImplementedError("TODO 4-3: w3 돌연변이를 구현하세요")
                    
                    if random.uniform(0, 1) < args.mutation_prob:
                        # YOUR CODE HERE
                        # 힌트: w4의 shape은 (10, 3)입니다
                        raise NotImplementedError("TODO 4-4: w4 돌연변이를 구현하세요")
                    
                    genomes.append(new_genome)
    
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Saving best genome...")
        best_genomes[0].save('models/snake_ga_interrupted.npz')
        print("Saved to models/snake_ga_interrupted.npz")
    
    # 최종 저장
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best fitness: {best_fitness_ever:.1f}")
    print(f"Best score: {best_score_ever}")
    best_genomes[0].save('models/snake_ga_best.npz')
    print("Saved to models/snake_ga_best.npz")
    
    # 최종 플롯 생성
    if len(training_logs) > 1:
        print("\nGenerating final training plot...")
        try:
            plot_training_progress(log_file, 'logs/training_plot_final.png')
        except Exception as e:
            print(f"[Warning] Final plot generation failed: {e}")
    
    print(f"\nLogs saved to: {log_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
