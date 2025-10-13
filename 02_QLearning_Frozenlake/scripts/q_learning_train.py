#!/usr/bin/env python3
"""
FrozenLake Q-Learning 훈련 스크립트
"""

import argparse
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime

# 부모 디렉토리를 경로에 추가 (utils 임포트용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import save_q_table, save_training_log, save_hyperparameters


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, 
                 discount_factor=0.95, epsilon_start=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995):
        """
        Q-Learning 에이전트 초기화
        
        Args:
            state_size: 환경의 상태 개수
            action_size: 환경의 행동 개수
            learning_rate: 학습률 (alpha)
            discount_factor: 할인 인수 (gamma)
            epsilon_start: 초기 탐험률
            epsilon_min: 최소 탐험률
            epsilon_decay: 탐험률 감쇠율
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # TODO: Q-table을 0으로 초기화하세요
        # 힌트: numpy의 zeros 함수를 사용하여 (state_size, action_size) 크기의 2D 배열을 만드세요
        # Q-table[state, action]은 특정 상태에서 특정 행동을 취했을 때의 기대 보상을 저장합니다
        #YOUR CODE HERE
        raise NotImplementedError("Q-table 초기화를 구현하세요")
        
    def choose_action(self, state, training=True):
        """
        epsilon-greedy 정책을 사용하여 행동 선택
        
        Args:
            state: 현재 상태
            training: 훈련 모드 여부 (epsilon-greedy 사용) 또는 평가 모드 (greedy)
            
        Returns:
            선택된 행동
        """
        # TODO: ε-greedy 정책을 구현하세요
        # 힌트 1: training 모드일 때, epsilon 확률로 탐험(exploration)을 수행합니다
        # 힌트 2: 탐험: np.random.choice()로 무작위 행동 선택
        # 힌트 3: 활용(exploitation): Q-table에서 현재 상태의 Q-값이 가장 높은 행동 선택
        # 힌트 4: 동점인 경우를 처리하려면 np.where()로 최대값인 행동들을 찾고 그 중 무작위로 선택
        #YOUR CODE HERE
        raise NotImplementedError("epsilon-greedy 정책을 구현하세요")
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Q-learning 업데이트 규칙을 사용하여 Q-table 업데이트
        
        Args:
            state: 현재 상태
            action: 취한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        # TODO: Q-learning 업데이트 규칙(벨만 방정식)을 구현하세요
        # Q-learning 공식: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        # 
        # 힌트 1: 현재 Q-값 가져오기: current_q = self.q_table[state, action]
        # 힌트 2: 타겟 Q-값 계산:
        #         - 에피소드가 끝났으면(done=True): target_q = reward
        #         - 에피소드가 계속되면: target_q = reward + gamma * max(Q(next_state, all_actions))
        # 힌트 3: Q-값 업데이트: Q(s,a) = current_q + alpha * (target_q - current_q)
        # 힌트 4: self.learning_rate는 α(alpha), self.discount_factor는 γ(gamma)입니다
        #YOUR CODE HERE
        raise NotImplementedError("Q-learning 업데이트 규칙을 구현하세요")
    
    def decay_epsilon(self):
        """탐험률 감소"""
        # TODO: epsilon 값을 감소시키세요
        # 힌트 1: epsilon을 epsilon_decay만큼 곱해서 감소시킵니다
        # 힌트 2: epsilon이 epsilon_min 아래로 내려가지 않도록 제한해야 합니다
        # 힌트 3: max() 함수를 사용하여 최소값을 보장하세요
        #YOUR CODE HERE
        raise NotImplementedError("epsilon decay를 구현하세요")


def train_q_learning(env, agent, num_episodes, max_steps_per_episode=100, 
                    log_interval=500, verbose=True):
    """
    Q-learning 에이전트 훈련
    
    Args:
        env: Gymnasium 환경
        agent: Q-learning 에이전트
        num_episodes: 훈련 에피소드 수
        max_steps_per_episode: 에피소드당 최대 스텝 수
        log_interval: 진행 상황 로깅 주기
        verbose: 진행 상황 출력 여부
        
    Returns:
        훈련 통계를 포함한 딕셔너리
    """
    # 훈련 통계
    episode_rewards = []
    episode_lengths = []
    success_episodes = []
    epsilon_values = []
    
    # 성공률 추적 윈도우
    success_window = []
    window_size = 100
    
    if verbose:
        print(f"{num_episodes} 에피소드 Q-learning 훈련 시작...")
        print(f"환경: {env.spec.id}")
        print(f"상태 공간: {agent.state_size}")
        print(f"행동 공간: {agent.action_size}")
        print()
    
    # 훈련 루프
    for episode in tqdm(range(num_episodes), desc="훈련 중"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps_per_episode):
            # 행동 선택
            action = agent.choose_action(state, training=True)
            
            # 행동 실행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Q-table 업데이트
            agent.update_q_table(state, action, reward, next_state, done)
            
            # 상태 및 통계 업데이트
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # epsilon 감소
        agent.decay_epsilon()
        
        # 통계 기록
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_episodes.append(1 if total_reward > 0 else 0)
        epsilon_values.append(agent.epsilon)
        
        # 성공 윈도우 업데이트
        success_window.append(1 if total_reward > 0 else 0)
        if len(success_window) > window_size:
            success_window.pop(0)
        
        # 주기적 로깅
        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            success_rate = np.mean(success_window) * 100
            print(f"에피소드 {episode + 1}/{num_episodes}")
            print(f"  평균 보상 (최근 {log_interval}): {avg_reward:.3f}")
            print(f"  성공률 (최근 {window_size}): {success_rate:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print()
    
    # 최종 통계
    if verbose:
        final_success_rate = np.mean(success_window) * 100
        overall_success_rate = np.mean(success_episodes) * 100
        print("훈련 완료!")
        print(f"최종 성공률 (최근 {window_size} 에피소드): {final_success_rate:.1f}%")
        print(f"전체 성공률: {overall_success_rate:.1f}%")
        print(f"최종 epsilon: {agent.epsilon:.3f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_episodes': success_episodes,
        'epsilon_values': epsilon_values,
        'num_episodes': num_episodes,
        'final_success_rate': np.mean(success_window) * 100,
        'overall_success_rate': np.mean(success_episodes) * 100
    }


def main():
    parser = argparse.ArgumentParser(description="FrozenLake Q-Learning 훈련")
    
    # 환경 파라미터
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4",
                       help="맵 크기 (기본값: 4x4)")
    parser.add_argument("--slippery", action="store_true", default=True,
                       help="미끄러운 표면 활성화 (기본값: True)")
    parser.add_argument("--no-slippery", action="store_false", dest="slippery",
                       help="미끄러운 표면 비활성화")
    parser.add_argument("--random-map", action="store_true",
                       help="기본 맵 대신 랜덤 맵 생성")
    
    # 훈련 파라미터
    parser.add_argument("--episodes", type=int, default=5000,
                       help="훈련 에피소드 수 (기본값: 5000)")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="에피소드당 최대 스텝 수 (기본값: 100)")
    
    # Q-learning 하이퍼파라미터
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="학습률 (기본값: 0.1)")
    parser.add_argument("--gamma", type=float, default=0.95,
                       help="할인 인수 (기본값: 0.95)")
    parser.add_argument("--eps-start", type=float, default=1.0,
                       help="초기 epsilon (기본값: 1.0)")
    parser.add_argument("--eps-min", type=float, default=0.01,
                       help="최소 epsilon (기본값: 0.01)")
    parser.add_argument("--eps-decay", type=float, default=0.995,
                       help="Epsilon 감쇠율 (기본값: 0.995)")
    
    # 출력 파라미터
    parser.add_argument("--save-path", type=str, default=None,
                       help="Q-table 저장 경로 (기본값: weights/q_table_{맵크기}[_random].npy)")
    parser.add_argument("--log-path", type=str, default=None,
                       help="훈련 로그 저장 경로 (기본값: weights/training_log_{맵크기}[_random].json)")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드 (기본값: 42)")
    parser.add_argument("--quiet", action="store_true",
                       help="훈련 진행 상황 출력 억제")
    
    args = parser.parse_args()
    
    # 기본 저장 경로 생성 (맵 크기와 랜덤 여부에 따라)
    if args.save_path is None:
        suffix = f"_{args.map.replace('x', 'x')}"
        if args.random_map:
            suffix += "_random"
        args.save_path = f"weights/q_table{suffix}.npy"
    
    if args.log_path is None:
        suffix = f"_{args.map.replace('x', 'x')}"
        if args.random_map:
            suffix += "_random"
        args.log_path = f"weights/training_log{suffix}.json"
    
    # 랜덤 시드 설정
    np.random.seed(args.seed)
    
    # 환경 생성
    if args.random_map:
        # 랜덤 맵 생성
        map_size = int(args.map.split('x')[0])
        random_map = generate_random_map(size=map_size, p=0.8)
        env = gym.make("FrozenLake-v1", 
                       desc=random_map,
                       is_slippery=args.slippery,
                       render_mode=None)
        print(f"랜덤 {args.map} 맵 사용")
        print("맵 레이아웃:")
        for row in random_map:
            print(row)
        print()
    else:
        env = gym.make("FrozenLake-v1", 
                       map_name=args.map, 
                       is_slippery=args.slippery,
                       render_mode=None)  # 훈련 중 렌더링 없음
    
    # 환경 차원 정보 가져오기
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # 에이전트 생성
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=args.alpha,
        discount_factor=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_min=args.eps_min,
        epsilon_decay=args.eps_decay
    )
    
    # 에이전트 훈련
    training_stats = train_q_learning(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        verbose=not args.quiet
    )
    
    # Q-table 저장
    save_q_table(agent.q_table, args.save_path)
    
    # 훈련 로그 저장
    save_training_log(training_stats, args.log_path)
    
    # 하이퍼파라미터 저장
    hyperparams = {
        'map_name': args.map,
        'is_slippery': args.slippery,
        'random_map': args.random_map,
        'num_episodes': args.episodes,
        'max_steps_per_episode': args.max_steps,
        'learning_rate': args.alpha,
        'discount_factor': args.gamma,
        'epsilon_start': args.eps_start,
        'epsilon_min': args.eps_min,
        'epsilon_decay': args.eps_decay,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat()
    }
    
    hyperparams_path = args.save_path.replace('.npy', '_hyperparams.json')
    save_hyperparameters(hyperparams, hyperparams_path)
    
    env.close()
    
    if not args.quiet:
        print(f"\n훈련이 성공적으로 완료되었습니다!")
        print(f"Q-table 저장 위치: {args.save_path}")
        print(f"훈련 로그 저장 위치: {args.log_path}")
        print(f"하이퍼파라미터 저장 위치: {hyperparams_path}")


if __name__ == "__main__":
    main()