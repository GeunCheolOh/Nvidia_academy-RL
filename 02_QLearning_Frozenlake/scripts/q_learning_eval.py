#!/usr/bin/env python3
"""
FrozenLake Q-Learning 평가 스크립트
"""

import argparse
import numpy as np
import gymnasium as gym
import pygame
import time
import sys
import os
from tqdm import tqdm

# 부모 디렉토리를 경로에 추가 (utils 임포트용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_q_table, load_hyperparameters


class QLearningEvaluator:
    def __init__(self, q_table, render_mode="human"):
        """
        Q-Learning 평가자 초기화
        
        Args:
            q_table: 훈련된 Q-table
            render_mode: 렌더링 모드 ("human", "rgb_array", 또는 None)
        """
        self.q_table = q_table
        self.render_mode = render_mode
        
    def choose_action(self, state):
        """
        greedy 정책을 사용하여 행동 선택 (탐험 없음)
        
        Args:
            state: 현재 상태
            
        Returns:
            Q-table에 따른 최선의 행동
        """
        # TODO: Q-table을 사용하여 최선의 행동을 선택하세요 (greedy 정책)
        # 힌트 1: self.q_table[state]로 현재 상태의 모든 Q-값을 가져옵니다
        # 힌트 2: np.max()로 최대 Q-값을 찾습니다
        # 힌트 3: np.where()로 최대 Q-값을 가진 행동들을 찾습니다
        # 힌트 4: 동점인 경우 np.random.choice()로 무작위 선택합니다
        #YOUR CODE HERE
        raise NotImplementedError("greedy 행동 선택을 구현하세요")
    
    def evaluate_episodes(self, env, num_episodes, max_steps_per_episode=100, verbose=True):
        """
        여러 에피소드에 대해 에이전트를 평가
        
        Args:
            env: Gymnasium 환경
            num_episodes: 평가 에피소드 수
            max_steps_per_episode: 에피소드당 최대 스텝 수
            verbose: 진행 상황 출력 여부
            
        Returns:
            평가 통계를 포함한 딕셔너리
        """
        episode_rewards = []
        episode_lengths = []
        success_episodes = []
        
        if verbose:
            print(f"{num_episodes} 에피소드에 대해 에이전트 평가 중...")
            print(f"환경: {env.spec.id}")
            print()
        
        # 평가 루프
        for episode in tqdm(range(num_episodes), desc="평가 중", disable=not verbose):
            # TODO: 환경을 리셋하고 초기 상태를 받으세요
            # 힌트: env.reset()은 (state, info) 튜플을 반환합니다
            #YOUR CODE HERE
            raise NotImplementedError("환경 리셋을 구현하세요")
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # 최선의 행동 선택 (greedy 정책)
                action = self.choose_action(state)
                
                # TODO: 행동을 실행하고 결과를 받으세요
                # 힌트 1: env.step(action)을 호출합니다
                # 힌트 2: 반환값: (next_state, reward, terminated, truncated, info)
                # 힌트 3: done = terminated or truncated로 에피소드 종료 여부 확인
                #YOUR CODE HERE
                raise NotImplementedError("환경 step을 구현하세요")
                
                # 통계 업데이트
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # 에피소드 통계 기록
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            success_episodes.append(1 if total_reward > 0 else 0)
        
        # 최종 통계 계산
        avg_reward = np.mean(episode_rewards)
        success_rate = np.mean(success_episodes) * 100
        avg_length = np.mean(episode_lengths)
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  에피소드: {num_episodes}")
            print(f"  평균 보상: {avg_reward:.3f}")
            print(f"  성공률: {success_rate:.1f}%")
            print(f"  평균 에피소드 길이: {avg_length:.1f} 스텝")
        
        return {
            'num_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_episodes': success_episodes,
            'average_reward': avg_reward,
            'success_rate': success_rate,
            'average_length': avg_length
        }
    
    def demonstrate_policy(self, env, num_episodes=3, step_delay=0.8, verbose=True, interactive=True):
        """
        학습된 정책을 시각적 렌더링으로 시연
        
        Args:
            env: Gymnasium 환경
            num_episodes: 시연할 에피소드 수
            step_delay: 스텝 간 지연 시간 (초)
            verbose: 스텝 정보 출력 여부
            interactive: 에피소드 간 사용자 입력 대기 여부
        """
        if self.render_mode not in ["human", "rgb_array"]:
            print("경고: 시연은 render_mode='human' 또는 'rgb_array'가 필요합니다")
            return
        
        print(f"🎮 에이전트 플레이 시연 시작!")
        print(f"📺 에피소드 수: {num_episodes}")
        print(f"⏱️  스텝 딜레이: {step_delay}초")
        print(f"🎯 목표: S(시작) → G(골) 도달, H(구멍) 피하기")
        print()
        
        successes = 0
        
        for episode in range(num_episodes):
            print(f"🎬 Episode {episode + 1}/{num_episodes}")
            print("=" * 40)
            
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            path = [state]  # 경로 추적
            
            if verbose:
                print(f"🏁 시작 위치: {state} (위치: {state//4}, {state%4})")
            
            # 초기 상태 렌더링
            if self.render_mode == "human":
                env.render()
                time.sleep(step_delay)
            
            max_steps = 100
            while steps < max_steps:
                # 현재 상태의 Q-값들 표시
                q_values = self.q_table[state]
                action = self.choose_action(state)
                
                # Action names for display
                action_names = ["⬅️ Left", "⬇️ Down", "➡️ Right", "⬆️ Up"]
                action_symbols = ["←", "↓", "→", "↑"]
                
                if verbose:
                    print(f"📍 Step {steps + 1}:")
                    print(f"   현재 상태: {state} (위치: {state//4}, {state%4})")
                    print(f"   Q-값들: {q_values}")
                    print(f"   선택한 행동: {action} ({action_names[action]})")
                
                # 행동 실행
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 경로에 추가
                path.append(next_state)
                
                # 상태 업데이트
                state = next_state
                total_reward += reward
                steps += 1
                
                # 렌더링
                if self.render_mode == "human":
                    env.render()
                    time.sleep(step_delay)
                
                if verbose:
                    print(f"   ➡️ 다음 상태: {state} (위치: {state//4}, {state%4})")
                    print(f"   🎁 보상: {reward}")
                    
                    if done:
                        if reward > 0:
                            print("   🎉 골 도달!")
                        else:
                            print("   💀 구멍에 빠짐!")
                    print()
                
                if done:
                    break
            
            # 에피소드 요약
            result = "✅ 성공" if total_reward > 0 else "❌ 실패"
            if total_reward > 0:
                successes += 1
                
            print(f"📊 에피소드 결과:")
            print(f"   결과: {result}")
            print(f"   총 보상: {total_reward}")
            print(f"   소요 스텝: {steps}")
            print(f"   이동 경로: {' → '.join(map(str, path))}")
            print(f"   현재까지 성공률: {successes}/{episode+1} ({successes/(episode+1)*100:.1f}%)")
            print()
            
            # 다음 에피소드로 진행 확인
            if episode < num_episodes - 1:
                if interactive:
                    input("⏸️  Press Enter to continue to next episode...")
                else:
                    time.sleep(2)  # 자동으로 2초 후 진행
                print()
        
        # 최종 요약
        print("🏆 시연 완료!")
        print(f"   총 성공률: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
        
    def watch_agent_play(self, env, num_episodes=5, step_delay=1.0, show_q_values=True):
        """
        에이전트의 플레이를 자세히 관찰하는 함수
        """
        print("👀 에이전트 플레이 관찰 모드")
        print("Q-값과 의사결정 과정을 자세히 보여줍니다.")
        print()
        
        for episode in range(num_episodes):
            print(f"🎯 관찰 에피소드 {episode + 1}/{num_episodes}")
            print("-" * 50)
            
            state, _ = env.reset()
            done = False
            step = 0
            
            while not done and step < 100:
                if show_q_values:
                    q_values = self.q_table[state]
                    print(f"📊 상태 {state}의 Q-값 분석:")
                    actions = ["Left", "Down", "Right", "Up"]
                    for i, (action, q_val) in enumerate(zip(actions, q_values)):
                        marker = "🏆" if q_val == np.max(q_values) else "  "
                        print(f"   {marker} {action:>5}: {q_val:6.3f}")
                
                action = self.choose_action(state)
                actions = ["⬅️", "⬇️", "➡️", "⬆️"]
                print(f"🎯 선택된 행동: {actions[action]}")
                
                if self.render_mode == "human":
                    env.render()
                    time.sleep(step_delay)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                print(f"📍 {state} → {next_state}, 보상: {reward}")
                
                state = next_state
                step += 1
                
                if done:
                    result = "🎉 성공!" if reward > 0 else "💀 실패"
                    print(f"🏁 게임 종료: {result}")
                
                print()
            
            if episode < num_episodes - 1:
                input("계속하려면 Enter를 누르세요...")
                print()


def main():
    parser = argparse.ArgumentParser(description="FrozenLake Q-Learning 평가")
    
    # 환경 파라미터
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4",
                       help="맵 크기 (기본값: 4x4)")
    parser.add_argument("--slippery", action="store_true", default=True,
                       help="미끄러운 표면 활성화 (기본값: True)")
    parser.add_argument("--no-slippery", action="store_false", dest="slippery",
                       help="미끄러운 표면 비활성화")
    
    # 평가 파라미터
    parser.add_argument("--episodes", type=int, default=100,
                       help="평가 에피소드 수 (default: 100)")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="에피소드당 최대 스텝 수 (default: 100)")
    
    # 모델 파라미터
    parser.add_argument("--load-path", type=str, default="weights/q_table_4x4.npy",
                       help="Q-table 로드 경로 (기본값: weights/q_table_4x4.npy)")
    
    # 렌더링 파라미터
    parser.add_argument("--render", choices=["human", "none"], default="none",
                       help="렌더링 모드 (기본값: none)")
    parser.add_argument("--demonstrate", action="store_true",
                       help="시각적 렌더링으로 정책 시연")
    parser.add_argument("--demo-episodes", type=int, default=3,
                       help="시연할 에피소드 수 (default: 3)")
    parser.add_argument("--step-delay", type=float, default=0.8,
                       help="시연 시 스텝 간 지연 시간 (기본값: 0.8)")
    parser.add_argument("--watch-mode", action="store_true",
                       help="상세한 Q-값 분석이 포함된 관찰 모드")
    parser.add_argument("--auto-play", action="store_true",
                       help="자동 재생 모드 (에피소드 간 사용자 입력 없음)")
    
    # 기타 파라미터
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드 (기본값: 42)")
    parser.add_argument("--quiet", action="store_true",
                       help="평가 진행 상황 출력 억제")
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    np.random.seed(args.seed)
    
    # TODO: 학습된 Q-table을 로드하세요
    # 힌트 1: utils.io의 load_q_table() 함수를 사용합니다
    # 힌트 2: args.load_path를 인자로 전달합니다
    # 힌트 3: FileNotFoundError 예외 처리를 해야 합니다
    # 힌트 4: 파일이 없으면 에러 메시지를 출력하고 sys.exit(1)로 종료합니다
    #YOUR CODE HERE
    raise NotImplementedError("Q-table 로드를 구현하세요")
    
    # 호환성 확인을 위해 하이퍼파라미터 로드 시도
    hyperparams_path = args.load_path.replace('.npy', '_hyperparams.json')
    try:
        hyperparams = load_hyperparameters(hyperparams_path)
        
        # 호환성 확인
        if hyperparams.get('map_name') != args.map:
            print(f"Warning: Loaded model was trained on {hyperparams.get('map_name')} "
                  f"but evaluating on {args.map}")
        if hyperparams.get('is_slippery') != args.slippery:
            print(f"Warning: Loaded model was trained with slippery={hyperparams.get('is_slippery')} "
                  f"but evaluating with slippery={args.slippery}")
    except FileNotFoundError:
        print("Warning: Could not load hyperparameters file. Proceeding with evaluation...")
    
    # 렌더 모드 결정
    render_mode = args.render if args.render != "none" else None
    if args.demonstrate and render_mode != "human":
        render_mode = "human"
        print("시연을 위해 render_mode를 'human'으로 설정")
    
    # 환경 생성
    env = gym.make("FrozenLake-v1", 
                   map_name=args.map, 
                   is_slippery=args.slippery,
                   render_mode=render_mode)
    
    # 평가자 생성
    evaluator = QLearningEvaluator(q_table, render_mode=render_mode)
    
    # 평가 실행
    if not args.quiet:
        print(f"로드된 Q-table 형태: {q_table.shape}")
        print(f"환경 상태 공간: {env.observation_space.n}")
        print(f"환경 행동 공간: {env.action_space.n}")
        print()
    
    # 표준 평가
    eval_stats = evaluator.evaluate_episodes(
        env=env,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        verbose=not args.quiet
    )
    
    # 시연 모드
    if args.demonstrate:
        print()
        if args.watch_mode:
            evaluator.watch_agent_play(
                env=env,
                num_episodes=args.demo_episodes,
                step_delay=args.step_delay,
                show_q_values=True
            )
        else:
            evaluator.demonstrate_policy(
                env=env,
                num_episodes=args.demo_episodes,
                step_delay=args.step_delay,
                verbose=not args.quiet,
                interactive=not args.auto_play
            )
    
    env.close()
    
    if not args.quiet:
        print("\nEvaluation completed!")


if __name__ == "__main__":
    main()