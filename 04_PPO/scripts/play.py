"""
학습된 MultiDiscrete 모델 평가 및 시각화
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from env.pikachu_env import PikachuVolleyballEnvMultiDiscrete
from training.self_play import mirror_action_multidiscrete
from agents.ppo import PPOAgentMultiDiscrete
from evaluation.renderer import PygameRenderer
from training import get_device


def main():
    parser = argparse.ArgumentParser(description="Pikachu Volleyball MultiDiscrete Agent Evaluation")
    parser.add_argument("--model", type=str, required=True, help="학습된 모델 경로")
    parser.add_argument("--episodes", type=int, default=5, help="플레이할 게임 수")
    parser.add_argument("--deterministic", action="store_true", help="결정적 정책 사용")
    parser.add_argument("--scale", type=int, default=2, help="화면 확대 배율")
    args = parser.parse_args()
    
    print("=" * 60)
    print("학습된 에이전트 평가 (MultiDiscrete)")
    print("=" * 60)
    
    # 디바이스
    device = get_device()
    
    # 환경
    env = PikachuVolleyballEnvMultiDiscrete(winning_score=15)
    
    # 에이전트 로드
    print(f"\n모델 로드 중: {args.model}")
    agent = PPOAgentMultiDiscrete(
        observation_dim=15,
        device=str(device),
    )
    agent.load(args.model)
    
    # 렌더러
    renderer = PygameRenderer(scale=args.scale)
    
    print(f"\n설정:")
    print(f"  에피소드: {args.episodes}")
    print(f"  정책: {'결정적' if args.deterministic else '확률적'}")
    print(f"  화면 배율: {args.scale}x")
    print(f"  행동 공간: MultiDiscrete [3, 3, 2]")
    print("\nESC 키를 눌러 종료하세요\n")
    
    # 통계
    total_games = 0
    wins_p1 = 0
    scores_p1 = []
    scores_p2 = []
    
    try:
        for episode in range(args.episodes):
            print(f"게임 {episode + 1}/{args.episodes} 시작...")
            
            (obs_p1, obs_p2), _ = env.reset()
            done = False
            frame_count = 0
            
            while not done:
                # 종료 체크
                if renderer.check_quit():
                    print("\n사용자가 종료했습니다.")
                    raise KeyboardInterrupt
                
                # 행동 선택
                action_p1, _, _ = agent.select_action(obs_p1, deterministic=args.deterministic)
                action_p2_mirrored, _, _ = agent.select_action(obs_p2, deterministic=args.deterministic)
                action_p2 = mirror_action_multidiscrete(action_p2_mirrored)
                
                # 환경 진행
                (obs_p1, obs_p2), (r_p1, r_p2), terminated, truncated, info = \
                    env.step((action_p1, action_p2))
                
                done = terminated or truncated
                frame_count += 1
                
                # 렌더링
                renderer.render(env.physics, info['score_p1'], info['score_p2'])
                
                # 득점 시 출력
                if r_p1 != 0:
                    print(f"  점수: {info['score_p1']} - {info['score_p2']}")
            
            # 게임 종료
            total_games += 1
            scores_p1.append(info['score_p1'])
            scores_p2.append(info['score_p2'])
            
            if info['score_p1'] > info['score_p2']:
                wins_p1 += 1
                winner = "Player 1"
            else:
                winner = "Player 2"
            
            print(f"게임 {episode + 1} 종료: {info['score_p1']}-{info['score_p2']} (승자: {winner}, {frame_count}프레임)\n")
            
            # 잠시 결과 표시
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n평가 중단")
    
    finally:
        renderer.close()
        env.close()
    
    # 통계 출력
    if total_games > 0:
        print("\n" + "=" * 60)
        print("평가 결과")
        print("=" * 60)
        print(f"총 게임 수: {total_games}")
        print(f"Player 1 승리: {wins_p1} ({wins_p1/total_games*100:.1f}%)")
        print(f"Player 2 승리: {total_games - wins_p1} ({(total_games-wins_p1)/total_games*100:.1f}%)")
        print(f"\n평균 점수:")
        print(f"  Player 1: {sum(scores_p1)/len(scores_p1):.1f}")
        print(f"  Player 2: {sum(scores_p2)/len(scores_p2):.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()

