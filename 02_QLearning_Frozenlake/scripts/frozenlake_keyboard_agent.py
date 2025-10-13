#!/usr/bin/env python3
"""
FrozenLake 키보드 에이전트 - 환경 탐색을 위한 수동 제어
조작법: 화살표 키로 이동, Space로 리셋, Q로 종료
"""

import pygame
import gymnasium as gym
import numpy as np
import sys
import time


class FrozenLakeKeyboardAgent:
    def __init__(self, map_name="4x4", is_slippery=True, render_mode="human"):
        """
        FrozenLake 환경을 키보드 제어로 초기화
        
        Args:
            map_name: "4x4" 또는 "8x8"
            is_slippery: 호수가 미끄러운지 여부
            render_mode: pygame 창을 위한 "human" 또는 헤드리스를 위한 "rgb_array"
        """
        # TODO: Gymnasium 환경을 생성하세요
        # 힌트 1: gym.make()를 사용하여 "FrozenLake-v1" 환경을 만듭니다
        # 힌트 2: map_name, is_slippery, render_mode를 인자로 전달하세요
        # 힌트 3: 생성된 환경을 self.env에 저장하세요
        #YOUR CODE HERE
        raise NotImplementedError("환경 생성을 구현하세요")
        
        # 행동 매핑
        self.action_map = {
            pygame.K_LEFT: 0,   # 왼쪽
            pygame.K_DOWN: 1,   # 아래  
            pygame.K_RIGHT: 2,  # 오른쪽
            pygame.K_UP: 3      # 위
        }
        
        # 에피소드 통계
        self.episode_count = 0
        self.total_reward = 0
        self.total_steps = 0
        self.success_count = 0
        
        # 게임 상태
        self.state = None
        self.done = False
        
        print("FrozenLake 키보드 에이전트")
        print("조작법:")
        print("  화살표 키: 이동 (↑↓←→)")
        print("  Space: 에피소드 리셋")
        print("  Q: 종료")
        print("  R: 현재 통계 보기")
        print()
        
    def reset(self):
        """환경 리셋 및 새 에피소드 시작"""
        if self.done and self.episode_count > 0:
            print(f"에피소드 {self.episode_count} 완료!")
            print(f"  총 보상: {self.total_reward}")
            print(f"  소요 스텝: {self.total_steps}")
            if self.total_reward > 0:
                print("  결과: 성공! ✓")
                self.success_count += 1
            else:
                print("  결과: 실패 ✗")
            print()
        
        # TODO: 환경을 리셋하고 초기 상태를 받으세요
        # 힌트 1: self.env.reset()을 호출하면 (state, info) 튜플을 반환합니다
        # 힌트 2: state는 에이전트의 현재 위치를 나타내는 정수입니다
        # 힌트 3: 반환된 state를 self.state에 저장하세요
        #YOUR CODE HERE
        raise NotImplementedError("환경 리셋을 구현하세요")
        
        self.done = False
        self.episode_count += 1
        self.total_reward = 0
        self.total_steps = 0
        
        print(f"에피소드 {self.episode_count} 시작")
        print(f"시작 위치: {self.state}")
        
    def step(self, action):
        """환경에서 행동 실행"""
        if self.done:
            print("에피소드 종료! Space를 눌러 리셋하세요.")
            return
            
        old_state = self.state
        
        # TODO: 환경에서 행동을 실행하고 결과를 받으세요
        # 힌트 1: self.env.step(action)을 호출합니다
        # 힌트 2: 반환값은 (next_state, reward, terminated, truncated, info) 튜플입니다
        #         - next_state: 다음 상태 (정수)
        #         - reward: 받은 보상 (0 또는 1)
        #         - terminated: 에피소드가 종료되었는지 (골 도달 또는 구멍)
        #         - truncated: 시간 초과로 종료되었는지
        # 힌트 3: self.state, reward, terminated, truncated를 적절히 저장하세요
        # 힌트 4: self.done = terminated or truncated로 설정하세요
        #YOUR CODE HERE
        raise NotImplementedError("환경 step을 구현하세요")
        
        self.total_reward += reward
        self.total_steps += 1
        
        # 화면 표시용 행동 이름
        action_names = ["왼쪽", "아래", "오른쪽", "위"]
        
        print(f"스텝 {self.total_steps}: {action_names[action]} | "
              f"상태: {old_state} → {self.state} | "
              f"보상: {reward} | "
              f"완료: {self.done}")
        
        if self.done:
            if reward > 0:
                print("🎉 골 도달!")
            else:
                print("💀 구멍에 빠지거나 시간 초과!")
    
    def show_statistics(self):
        """현재 통계 표시"""
        if self.episode_count == 0:
            print("아직 완료된 에피소드가 없습니다.")
            return
            
        success_rate = (self.success_count / self.episode_count) * 100
        print()
        print("=== 통계 ===")
        print(f"완료된 에피소드: {self.episode_count}")
        print(f"성공: {self.success_count}")
        print(f"성공률: {success_rate:.1f}%")
        print("==========")
        print()
    
    def run(self):
        """pygame 이벤트 처리를 포함한 메인 게임 루프"""
        pygame.init()
        
        # human 렌더 모드 사용 시 디스플레이 초기화
        if self.env.render_mode == "human":
            # 환경이 디스플레이 생성을 처리합니다
            pass
        else:
            # 이벤트 처리를 위한 작은 창 생성
            screen = pygame.display.set_mode((400, 300))
            pygame.display.set_caption("FrozenLake Keyboard Control")
        
        clock = pygame.time.Clock()
        
        # 첫 번째 에피소드 시작
        self.reset()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    
                    elif event.key == pygame.K_SPACE:
                        self.reset()
                    
                    elif event.key == pygame.K_r:
                        self.show_statistics()
                    
                    elif event.key in self.action_map:
                        action = self.action_map[event.key]
                        self.step(action)
            
            # TODO: 환경을 화면에 렌더링하세요
            # 힌트 1: self.env.render_mode가 "human"일 때만 렌더링합니다
            # 힌트 2: self.env.render()를 호출하여 화면을 업데이트하세요
            #YOUR CODE HERE
            
            clock.tick(60)  # 60 FPS
        
        # 최종 통계
        if self.done and self.episode_count > 0:
            # 최종 에피소드 통계 처리
            if self.total_reward > 0:
                self.success_count += 1
        
        self.show_statistics()
        print("게임 종료. 안녕히가세요!")
        
        self.env.close()
        pygame.quit()


def main():
    """커맨드 라인 인자를 포함한 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FrozenLake 키보드 에이전트")
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4",
                       help="맵 크기 (기본값: 4x4)")
    parser.add_argument("--slippery", action="store_true", default=False,
                       help="미끄러운 표면 활성화 (키보드 플레이 기본값: False)")
    parser.add_argument("--no-slippery", action="store_false", dest="slippery",
                       help="미끄러운 표면 비활성화")
    parser.add_argument("--headless", action="store_true",
                       help="헤드리스 모드로 실행 (pygame 창 없음)")
    
    args = parser.parse_args()
    
    render_mode = "rgb_array" if args.headless else "human"
    
    print(f"FrozenLake {args.map} 시작 (slippery: {args.slippery})")
    
    try:
        agent = FrozenLakeKeyboardAgent(
            map_name=args.map,
            is_slippery=args.slippery,
            render_mode=render_mode
        )
        agent.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()