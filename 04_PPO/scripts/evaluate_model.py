"""
Best Model 통계 평가 (렌더링 없이)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from env.pikachu_env import PikachuVolleyballEnvMultiDiscrete
from training.self_play import mirror_action_multidiscrete
from agents.ppo import PPOAgentMultiDiscrete
from training import get_device


def main():
    parser = argparse.ArgumentParser(description="Best Model 통계 평가")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_model.pth", help="모델 경로")
    parser.add_argument("--episodes", type=int, default=100, help="평가 에피소드 수")
    parser.add_argument("--deterministic", action="store_true", help="결정적 정책 사용")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Best Model 통계 평가")
    print("=" * 60)
    
    device = get_device()
    env = PikachuVolleyballEnvMultiDiscrete(winning_score=15)
    
    print(f"\n모델 로드: {args.model}")
    agent = PPOAgentMultiDiscrete(observation_dim=15, device=str(device))
    agent.load(args.model)
    
    print(f"평가 에피소드: {args.episodes}")
    print(f"정책: {'결정적' if args.deterministic else '확률적'}\n")
    
    scores_p1 = []
    scores_p2 = []
    wins_p1 = 0
    episode_lengths = []
    
    for ep in range(args.episodes):
        (obs_p1, obs_p2), _ = env.reset()
        done = False
        length = 0
        
        while not done:
            action_p1, _, _ = agent.select_action(obs_p1, deterministic=args.deterministic)
            action_p2_mirrored, _, _ = agent.select_action(obs_p2, deterministic=args.deterministic)
            action_p2 = mirror_action_multidiscrete(action_p2_mirrored)
            
            (obs_p1, obs_p2), _, terminated, truncated, info = env.step((action_p1, action_p2))
            done = terminated or truncated
            length += 1
        
        scores_p1.append(info['score_p1'])
        scores_p2.append(info['score_p2'])
        episode_lengths.append(length)
        if info['score_p1'] > info['score_p2']:
            wins_p1 += 1
        
        if (ep + 1) % 10 == 0:
            print(f"진행: {ep + 1}/{args.episodes} 에피소드 완료")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"총 게임 수: {args.episodes}")
    print(f"\n점수 통계:")
    print(f"  Player 1 평균: {np.mean(scores_p1):.2f} ± {np.std(scores_p1):.2f}")
    print(f"  Player 2 평균: {np.mean(scores_p2):.2f} ± {np.std(scores_p2):.2f}")
    print(f"  점수 차이: {np.mean(scores_p1) - np.mean(scores_p2):.2f}")
    print(f"\n승률:")
    print(f"  Player 1: {wins_p1}/{args.episodes} ({wins_p1/args.episodes*100:.1f}%)")
    print(f"  Player 2: {args.episodes-wins_p1}/{args.episodes} ({(args.episodes-wins_p1)/args.episodes*100:.1f}%)")
    print(f"\n게임 길이:")
    print(f"  평균 프레임: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
    print(f"  최소/최대: {min(episode_lengths)} / {max(episode_lengths)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

