"""
Manual Play - 키보드로 뱀게임 플레이
방향키로 조작, ESC로 종료, SPACE로 일시정지
"""
import pygame
from snake_game import SnakeGame, SCREEN_SIZE, PIXEL_SIZE


def main():
    """메인 함수"""
    print("=" * 60)
    print("Snake Game - Manual Play")
    print("=" * 60)
    print("Controls:")
    print("  Arrow Keys - Move")
    print("  SPACE - Pause/Resume")
    print("  ESC - Quit")
    print("=" * 60)
    
    # Pygame 초기화
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_SIZE * PIXEL_SIZE, 
                                     SCREEN_SIZE * PIXEL_SIZE))
    pygame.display.set_caption('Snake - Manual Play')
    
    game_count = 0
    
    while True:
        game_count += 1
        print(f"\nGame #{game_count}")
        
        # 게임 생성 및 실행
        game = SnakeGame(screen, genome=None)
        fitness, score = game.run(manual=True, training_mode=False)
        
        print(f"  Score: {score}")
        print(f"  Fitness: {fitness:.1f}")


if __name__ == '__main__':
    main()

