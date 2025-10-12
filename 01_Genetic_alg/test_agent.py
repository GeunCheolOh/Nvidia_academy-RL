"""
Test Agent - 학습된 에이전트 테스트 및 렌더링
"""
import pygame
import argparse
import os
from snake_game import SnakeGame, SCREEN_SIZE, PIXEL_SIZE
from genome import Genome


def main():
    parser = argparse.ArgumentParser(description='Test trained Snake AI agent')
    parser.add_argument('--model', type=str, default='models/snake_ga_best.npz',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to test')
    parser.add_argument('--fps', type=int, default=60,
                       help='Game speed (FPS)')
    args = parser.parse_args()
    
    # 모델 존재 확인
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("Train a model first using train_ga.py")
        return
    
    print("=" * 60)
    print("Testing Trained Agent")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"FPS: {args.fps}")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  ESC - Quit")
    print("=" * 60)
    
    # Pygame 초기화
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_SIZE * PIXEL_SIZE, 
                                     SCREEN_SIZE * PIXEL_SIZE))
    pygame.display.set_caption('Snake - Testing Agent')
    
    # 모델 로드
    genome = Genome()
    genome.load(args.model)
    print(f"\n[OK] Model loaded (fitness: {genome.fitness:.1f})")
    
    # 게임 속도 조절을 위해 임시로 FPS 변경
    import snake_game
    original_fps = snake_game.FPS
    snake_game.FPS = args.fps
    
    scores = []
    fitnesses = []
    
    try:
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            
            game = SnakeGame(screen, genome=genome)
            fitness, score = game.run(manual=False, training_mode=False)
            
            scores.append(score)
            fitnesses.append(fitness)
            
            print(f"  Score: {score}")
            print(f"  Fitness: {fitness:.1f}")
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        # FPS 원상복구
        snake_game.FPS = original_fps
    
    # 통계 출력
    if scores:
        import numpy as np
        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)
        print(f"Episodes completed: {len(scores)}")
        print(f"Average score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"Best score: {max(scores)}")
        print(f"Average fitness: {np.mean(fitnesses):.1f}")
        print("=" * 60)


if __name__ == '__main__':
    main()

