#!/usr/bin/env python3
"""
학습된 DQN 에이전트 시연

이 스크립트는 학습된 DQN 에이전트가 CarRacing을 플레이하는 것을 보여줍니다.
저장된 모델 가중치를 로드하고 에이전트의 성능을 무작위 행동과 비교합니다.

사용법:
    python demo_trained_agent.py [--model model.pth] [--episodes 5] [--compare]

작성자: DQN Racing Tutorial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import cv2
import argparse
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List, Dict
import pygame


# 학습 스크립트에서 import
import sys
sys.path.append(str(Path(__file__).parent.parent / "training"))
from dqn_training import DQN, CarRacingWrapper, HYPERPARAMETERS


# ============================================================================
# 에이전트 데모 클래스
# ============================================================================

class DQNDemo:
    """학습된 DQN 에이전트를 위한 데모 클래스"""
    
    def __init__(self, model_path: Optional[str] = None, render: bool = True):
        """
        데모 환경 초기화
        
        Args:
            model_path: 학습된 모델 가중치 경로
            render: 환경 렌더링 여부
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {self.device}")
        
        # 환경 초기화
        render_mode = "human" if render else "rgb_array"
        self.env = CarRacingWrapper(render_mode=render_mode)
        
        # 네트워크 초기화
        self.network = DQN(action_dim=4).to(self.device)
        
        # 모델 로드 (제공된 경우)
        self.model_loaded = False
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._find_and_load_best_model()
            
        # 데모 통계
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = []
        
    def load_model(self, model_path: str):
        """학습된 모델 가중치 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 다양한 체크포인트 형식 처리
            if 'main_network' in checkpoint:
                state_dict = checkpoint['main_network']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            self.network.load_state_dict(state_dict)
            self.network.eval()  # 평가 모드로 설정
            
            print(f"✓ 모델 로드 완료: {model_path}")
            self.model_loaded = True
            
        except Exception as e:
            print(f"✗ 모델 로드 실패: {e}")
            print("무작위 에이전트를 대신 사용합니다")
            self.model_loaded = False
            
    def _find_and_load_best_model(self):
        """사용 가능한 최고의 모델을 찾아서 로드"""
        models_dir = Path(__file__).parent.parent / "models" / "saved_weights"
        
        # 최고 모델 찾기
        best_model = models_dir / "dqn_best.pth"
        if best_model.exists():
            self.load_model(str(best_model))
            return
            
        # 최종 모델 찾기
        final_model = models_dir / "dqn_final.pth"
        if final_model.exists():
            self.load_model(str(final_model))
            return
            
        # 아무 모델이나 찾기
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            if model_files:
                # 수정 시간으로 정렬하여 가장 최근 것 선택
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                self.load_model(str(latest_model))
                return
                
        print("⚠️  학습된 모델을 찾을 수 없습니다!")
        print("먼저 모델을 학습시키세요:")
        print("python training/dqn_training.py")
        
    def select_action(self, state: np.ndarray, use_model: bool = True) -> int:
        """
        학습된 모델 또는 무작위 정책으로 행동 선택
        
        Args:
            state: 현재 상태
            use_model: 학습된 모델 사용 여부
            
        Returns:
            선택된 행동
            
        TODO: 학습된 DQN 모델로 행동을 선택하세요
        힌트 1: use_model이 True이고 모델이 로드되었다면, 모델을 사용합니다
        힌트 2: torch.no_grad()로 그래디언트 계산을 비활성화합니다
        힌트 3: 상태를 텐서로 변환하고 배치 차원을 추가합니다 (unsqueeze(0))
        힌트 4: 네트워크로 Q-값을 계산하고 argmax()로 최대값의 인덱스를 반환합니다
        힌트 5: use_model이 False이거나 모델이 없으면 무작위 행동을 반환합니다
        """
        #YOUR CODE HERE
        raise NotImplementedError("학습된 모델로 행동 선택을 구현하세요")
            
    def run_episode(self, use_model: bool = True, max_steps: int = 1000) -> Tuple[float, int, List[int]]:
        """
        에이전트로 단일 에피소드 실행
        
        Args:
            use_model: 학습된 모델 사용 여부
            max_steps: 에피소드당 최대 스텝 수
            
        Returns:
            (총_보상, 에피소드_길이, 수행한_행동들) 튜플
        """
        state = self.env.reset()
        total_reward = 0.0
        actions_taken = []
        
        for step in range(max_steps):
            action = self.select_action(state, use_model)
            actions_taken.append(action)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
                
        return total_reward, step + 1, actions_taken
        
    def demo_single_agent(self, num_episodes: int = 5, use_model: bool = True):
        """
        여러 에피소드 동안 단일 에이전트 데모
        
        Args:
            num_episodes: 실행할 에피소드 수
            use_model: 학습된 모델 사용 여부
        """
        agent_type = "학습된 DQN" if (use_model and self.model_loaded) else "무작위"
        print(f"\n{'='*60}")
        print(f"{agent_type.upper()} 에이전트 시연")
        print(f"{'='*60}")
        
        episode_rewards = []
        episode_lengths = []
        all_actions = []
        
        for episode in range(num_episodes):
            print(f"\n에피소드 {episode + 1}/{num_episodes}")
            print("-" * 30)
            
            start_time = time.time()
            reward, length, actions = self.run_episode(use_model)
            episode_time = time.time() - start_time
            
            episode_rewards.append(reward)
            episode_lengths.append(length)
            all_actions.extend(actions)
            
            print(f"보상: {reward:8.2f}")
            print(f"길이: {length:4d} 스텝")
            print(f"시간: {episode_time:6.2f}초")
            
            # 이 에피소드의 행동 분포
            action_counts = np.bincount(actions, minlength=4)
            action_names = ['왼쪽', '직진', '오른쪽', '브레이크']
            print("행동:", end="")
            for i, (name, count) in enumerate(zip(action_names, action_counts)):
                print(f" {name}: {count:3d}", end="")
            print()
            
        # 요약 통계
        print(f"\n{agent_type} 에이전트 요약:")
        print("-" * 30)
        print(f"에피소드:      {num_episodes}")
        print(f"평균 보상:     {np.mean(episode_rewards):8.2f}")
        print(f"표준편차:      {np.std(episode_rewards):8.2f}")
        print(f"최고 보상:     {np.max(episode_rewards):8.2f}")
        print(f"최저 보상:     {np.min(episode_rewards):8.2f}")
        print(f"평균 길이:     {np.mean(episode_lengths):6.1f} 스텝")
        
        # 전체 행동 분포
        total_action_counts = np.bincount(all_actions, minlength=4)
        print(f"행동 분포:")
        for i, (name, count) in enumerate(zip(action_names, total_action_counts)):
            percentage = count / len(all_actions) * 100
            print(f"  {name:8}: {count:5d} ({percentage:5.1f}%)")
            
        return episode_rewards, episode_lengths
        
    def compare_agents(self, num_episodes: int = 5):
        """
        학습된 에이전트 vs 무작위 에이전트 비교
        
        Args:
            num_episodes: 에이전트당 에피소드 수
        """
        if not self.model_loaded:
            print("⚠️  비교할 학습된 모델이 없습니다!")
            print("무작위 에이전트 데모만 실행합니다...")
            self.demo_single_agent(num_episodes, use_model=False)
            return
            
        print(f"\n{'='*60}")
        print("에이전트 비교")
        print(f"{'='*60}")
        
        # 학습된 에이전트 실행
        print(f"\n🤖 학습된 DQN 에이전트 테스트...")
        trained_rewards, trained_lengths = self.demo_single_agent(num_episodes, use_model=True)
        
        # 무작위 에이전트 실행
        print(f"\n🎲 무작위 에이전트 테스트...")
        random_rewards, random_lengths = self.demo_single_agent(num_episodes, use_model=False)
        
        # 통계적 비교
        print(f"\n{'='*60}")
        print("비교 결과")
        print(f"{'='*60}")
        
        print(f"{'지표':<20} {'학습된 DQN':<15} {'무작위':<15} {'개선도':<15}")
        print("-" * 65)
        
        # 보상
        trained_avg = np.mean(trained_rewards)
        random_avg = np.mean(random_rewards)
        reward_improvement = ((trained_avg - random_avg) / abs(random_avg)) * 100
        
        print(f"{'평균 보상':<20} {trained_avg:<15.2f} {random_avg:<15.2f} {reward_improvement:<15.1f}%")
        
        # 에피소드 길이
        trained_len_avg = np.mean(trained_lengths)
        random_len_avg = np.mean(random_lengths)
        length_improvement = ((trained_len_avg - random_len_avg) / random_len_avg) * 100
        
        print(f"{'평균 길이':<20} {trained_len_avg:<15.1f} {random_len_avg:<15.1f} {length_improvement:<15.1f}%")
        
        # 최고 성능
        trained_best = np.max(trained_rewards)
        random_best = np.max(random_rewards)
        best_improvement = ((trained_best - random_best) / abs(random_best)) * 100
        
        print(f"{'최고 보상':<20} {trained_best:<15.2f} {random_best:<15.2f} {best_improvement:<15.1f}%")
        
        # 일관성 (낮은 표준편차가 더 좋음)
        trained_std = np.std(trained_rewards)
        random_std = np.std(random_rewards)
        consistency_improvement = ((random_std - trained_std) / random_std) * 100
        
        print(f"{'일관성':<20} {trained_std:<15.2f} {random_std:<15.2f} {consistency_improvement:<15.1f}%")
        
        # 통계적 유의성 검정
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_ind(trained_rewards, random_rewards)
            print(f"\n통계적 검정 (t-test):")
            print(f"  t-통계량: {t_stat:.3f}")
            print(f"  p-값:     {p_value:.4f}")
            if p_value < 0.05:
                print("  결과: 통계적으로 유의한 차이가 있습니다!")
            else:
                print("  결과: 유의한 차이 없음 (더 많은 학습이 필요합니다)")
        except ImportError:
            print("\n통계적 유의성 검정을 위해 scipy를 설치하세요")
            
        # 비교 그래프 생성
        self._plot_comparison(trained_rewards, random_rewards)
        
    def _plot_comparison(self, trained_rewards: List[float], random_rewards: List[float]):
        """비교 그래프 생성"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 에피소드 보상 비교
            episodes = range(1, len(trained_rewards) + 1)
            ax1.plot(episodes, trained_rewards, 'b-o', label='학습된 DQN', linewidth=2)
            ax1.plot(episodes, random_rewards, 'r-s', label='무작위', linewidth=2)
            ax1.set_xlabel('에피소드')
            ax1.set_ylabel('보상')
            ax1.set_title('에피소드 보상 비교')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot 비교
            ax2.boxplot([trained_rewards, random_rewards], 
                       labels=['학습된 DQN', '무작위'])
            ax2.set_ylabel('보상')
            ax2.set_title('보상 분포 비교')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 그래프 저장
            logs_dir = Path(__file__).parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            plot_path = logs_dir / "agent_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n비교 그래프 저장: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"그래프 생성 실패: {e}")
            
    def interactive_demo(self):
        """사용자 조작이 가능한 대화형 데모"""
        if not self.model_loaded:
            print("⚠️  학습된 모델이 없습니다!")
            return
            
        print(f"\n{'='*60}")
        print("대화형 데모")
        print(f"{'='*60}")
        print("조작법:")
        print("  SPACE - 학습된/무작위 에이전트 전환")
        print("  R     - 에피소드 리셋")
        print("  ESC   - 종료")
        print("  P     - 일시정지/재개")
        print("-" * 60)
        
        pygame.init()
        clock = pygame.time.Clock()
        
        use_model = True
        paused = False
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        use_model = not use_model
                        agent_type = "학습된 DQN" if use_model else "무작위"
                        print(f"전환됨: {agent_type}")
                    elif event.key == pygame.K_r:
                        state = self.env.reset()
                        episode_reward = 0.0
                        episode_steps = 0
                        print("에피소드 리셋")
                    elif event.key == pygame.K_p:
                        paused = not paused
                        print("일시정지" if paused else "재개")
                        
            if not paused:
                action = self.select_action(state, use_model)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # 정보 표시
                agent_type = "DQN" if use_model else "무작위"
                action_names = ['왼쪽', '직진', '오른쪽', '브레이크']
                print(f"\r{agent_type} | 스텝: {episode_steps:4d} | "
                      f"보상: {episode_reward:7.2f} | "
                      f"행동: {action_names[action]}", end="", flush=True)
                
                if terminated or truncated:
                    print(f"\n에피소드 종료! 최종 보상: {episode_reward:.2f}")
                    state = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
                    
            clock.tick(30)  # 30 FPS
            
        pygame.quit()
        print("\n대화형 데모 종료")
        
    def cleanup(self):
        """리소스 정리"""
        self.env.close()


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """에이전트 데모를 위한 메인 함수"""
    parser = argparse.ArgumentParser(description='DQN 에이전트 데모')
    parser.add_argument('--model', type=str, default=None,
                       help='학습된 모델 파일 경로')
    parser.add_argument('--episodes', type=int, default=5,
                       help='실행할 에피소드 수')
    parser.add_argument('--compare', action='store_true',
                       help='학습된 에이전트 vs 무작위 에이전트 비교')
    parser.add_argument('--interactive', action='store_true',
                       help='대화형 데모 실행')
    parser.add_argument('--no-render', action='store_true',
                       help='렌더링 비활성화')
    
    args = parser.parse_args()
    
    print("DQN 에이전트 데모")
    print("=" * 60)
    
    # 데모 생성
    demo = DQNDemo(model_path=args.model, render=not args.no_render)
    
    try:
        if args.interactive:
            demo.interactive_demo()
        elif args.compare:
            demo.compare_agents(args.episodes)
        else:
            demo.demo_single_agent(args.episodes, use_model=True)
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 데모가 중단되었습니다")
    except Exception as e:
        print(f"데모 실행 오류: {e}")
    finally:
        demo.cleanup()
        
    print("\n데모 완료!")


if __name__ == "__main__":
    main()
