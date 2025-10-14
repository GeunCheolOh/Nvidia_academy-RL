#!/usr/bin/env python3
"""
수동 CarRacing 게임 테스트

이 스크립트는 키보드 컨트롤을 사용하여 CarRacing 게임을 수동으로 플레이할 수 있게 합니다.
AI 에이전트를 구현하기 전에 게임 환경을 이해하는 데 유용합니다.

조작법:
    방향키:
        ↑ (위)    - 가속
        ↓ (아래)  - 브레이크
        ← (왼쪽)  - 왼쪽으로 조향
        → (오른쪽) - 오른쪽으로 조향
    
    기타 키:
        ESC       - 게임 종료
        R         - 에피소드 리셋
        SPACE     - 일시정지/재개

작성자: DQN Racing Tutorial
"""

import gymnasium as gym
import pygame
import numpy as np
import sys
import time
from typing import Tuple, Dict, Any


class ManualCarRacing:
    """CarRacing 환경을 위한 수동 조작 인터페이스"""
    
    def __init__(self, render_mode: str = "human"):
        """
        수동 레이싱 환경 초기화
        
        Args:
            render_mode: 환경의 렌더링 모드
        """
        self.env = None
        self.render_mode = render_mode
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.paused = False
        
        # 게임 통계
        self.episode_count = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.max_reward = float('-inf')
        self.episode_rewards = []
        
        # 조작 상태
        self.action = np.array([0.0, 0.0, 0.0])  # [조향, 가스, 브레이크] 또는 이산 행동
        self.keys_pressed = set()
        
        print("수동 CarRacing 게임")
        print("=" * 40)
        print("조작법:")
        print("  ↑ (위)    - 가속")
        print("  ↓ (아래)  - 브레이크") 
        print("  ← (왼쪽)  - 왼쪽으로 조향")
        print("  → (오른쪽) - 오른쪽으로 조향")
        print("  ESC       - 종료")
        print("  R         - 리셋")
        print("  SPACE     - 일시정지/재개")
        print("=" * 40)
        
    def init_environment(self):
        """레이싱 환경 초기화 (CarRacing 또는 대체 환경)"""
        # 먼저 CarRacing 시도
        try:
            self.env = gym.make('CarRacing-v3', render_mode=self.render_mode)
            self.env_name = "CarRacing-v3"
            print("✓ CarRacing 환경 초기화 성공")
            return True
        except Exception as e:
            print(f"⚠️  CarRacing을 사용할 수 없습니다: {e}")
            
        # CarRacing이 실패하면 CartPole로 대체
        try:
            self.env = gym.make('CartPole-v1', render_mode=self.render_mode)
            self.env_name = "CartPole-v1"
            print("✓ 대체 환경으로 CartPole 사용")
            print("  (CarRacing은 Box2D가 필요합니다: pip install 'gymnasium[box2d]')")
            return True
        except Exception as e:
            print(f"✗ 환경 초기화 실패: {e}")
            return False
            
    def reset_episode(self):
        """새 에피소드를 위해 환경 리셋"""
        if self.env is None:
            return None
            
        try:
            obs, info = self.env.reset()
            
            # 통계 업데이트
            if self.total_reward > 0:
                self.episode_rewards.append(self.total_reward)
                if self.total_reward > self.max_reward:
                    self.max_reward = self.total_reward
                    
            self.episode_count += 1
            self.total_reward = 0.0
            self.step_count = 0
            self.action = np.array([0.0, 0.0, 0.0])
            
            print(f"\n--- 에피소드 {self.episode_count} 시작 ---")
            return obs
            
        except Exception as e:
            print(f"환경 리셋 오류: {e}")
            return None
            
    def process_keyboard_input(self) -> bool:
        """
        키보드 입력을 처리하고 행동 업데이트
        
        Returns:
            bool: 종료가 요청되면 False, 그렇지 않으면 True
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
                    print("게임 일시정지" if self.paused else "게임 재개")
                    
            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)
                
        # 현재 눌린 키를 기반으로 행동 업데이트
        self.update_action_from_keys()
        return True
        
    def update_action_from_keys(self):
        """
        현재 눌린 키를 기반으로 행동 배열 업데이트
        
        TODO: 키보드 입력을 환경 행동으로 변환하세요
        힌트 1: CartPole은 이산 행동 (0=왼쪽, 1=오른쪽)을 사용합니다
        힌트 2: CarRacing은 연속 행동 [조향, 가스, 브레이크]를 사용합니다
        힌트 3: 조향: 왼쪽=-1.0, 오른쪽=1.0, 중립=0.0
        힌트 4: 가스/브레이크: 눌림=1.0, 안눌림=0.0
        """
        #YOUR CODE HERE
        raise NotImplementedError("키보드 입력을 행동으로 변환하세요")
            
    def display_info(self):
        """콘솔에 게임 정보 표시"""
        if self.step_count % 60 == 0:  # 1초마다 업데이트 (60 FPS)
            if hasattr(self, 'env_name') and 'CartPole' in self.env_name:
                action_names = ['왼쪽으로 밀기', '오른쪽으로 밀기']
                action_str = action_names[self.action] if self.action < len(action_names) else str(self.action)
            else:
                action_str = f"[{self.action[0]:.1f}, {self.action[1]:.1f}, {self.action[2]:.1f}]"
                
            info_str = (
                f"에피소드: {self.episode_count} | "
                f"스텝: {self.step_count} | "
                f"보상: {self.total_reward:.1f} | "
                f"행동: {action_str}"
            )
            print(f"\r{info_str}", end="", flush=True)
            
    def display_statistics(self):
        """게임 통계 표시"""
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards)
            print(f"\n\n게임 통계:")
            print(f"  완료한 에피소드: {len(self.episode_rewards)}")
            print(f"  평균 보상: {avg_reward:.2f}")
            print(f"  최고 보상: {self.max_reward:.2f}")
            print(f"  마지막 보상: {self.episode_rewards[-1]:.2f}")
        else:
            print(f"\n총 스텝: {self.step_count}")
            print(f"현재 보상: {self.total_reward:.2f}")
            
    def run(self):
        """수동 레이싱 게임 실행"""
        if not self.init_environment():
            return
            
        pygame.init()
        
        # 첫 에피소드 시작
        obs = self.reset_episode()
        if obs is None:
            return
            
        print("게임 시작! 방향키로 자동차를 조작하세요.")
        running = True
        
        try:
            while running:
                # 입력 처리
                running = self.process_keyboard_input()
                if not running:
                    break
                    
                # 일시정지 상태면 게임 로직 건너뛰기
                if self.paused:
                    self.clock.tick(10)  # 일시정지 시 낮은 FPS
                    continue
                    
                # 환경에서 행동 수행
                try:
                    obs, reward, terminated, truncated, info = self.env.step(self.action)
                    
                    # 통계 업데이트
                    self.total_reward += reward
                    self.step_count += 1
                    
                    # 정보 표시
                    self.display_info()
                    
                    # 에피소드 종료 확인
                    if terminated or truncated:
                        print(f"\n에피소드 {self.episode_count} 종료!")
                        print(f"최종 보상: {self.total_reward:.2f}")
                        print("리셋하려면 'R' 또는 종료하려면 ESC를 누르세요")
                        
                except Exception as e:
                    print(f"\n스텝 실행 오류: {e}")
                    break
                    
                # FPS 조절
                self.clock.tick(self.fps)
                
        except KeyboardInterrupt:
            print("\n사용자에 의해 게임이 중단되었습니다")
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """리소스 정리"""
        self.display_statistics()
        
        if self.env:
            self.env.close()
            
        pygame.quit()
        print("\n플레이해 주셔서 감사합니다!")


def main():
    """수동 레이싱 게임을 실행하는 메인 함수"""
    print("수동 CarRacing 게임 초기화 중...")
    
    # pygame 사용 가능 여부 확인
    try:
        import pygame
    except ImportError:
        print("❌ Pygame을 찾을 수 없습니다. 설치: pip install pygame")
        sys.exit(1)
        
    # gymnasium 사용 가능 여부 확인
    try:
        import gymnasium as gym
    except ImportError:
        print("❌ Gymnasium을 찾을 수 없습니다. 설치: pip install gymnasium[classic_control]")
        sys.exit(1)
        
    # 게임 생성 및 실행
    game = ManualCarRacing(render_mode="human")
    game.run()


if __name__ == "__main__":
    main()
