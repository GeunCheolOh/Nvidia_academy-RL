"""
게임 시각화 데모

랜덤 에이전트가 플레이하는 것을 시각적으로 보여줍니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import PikachuVolleyballEnv
from evaluation.renderer import PygameRenderer


def main():
    """랜덤 플레이 데모"""
    print("=" * 60)
    print("Pikachu Volleyball 시각화 데모")
    print("=" * 60)
    print("\nESC 키를 눌러 종료하세요")
    print("랜덤 에이전트가 플레이합니다...\n")
    
    # 환경 생성 (15점제)
    env = PikachuVolleyballEnv(winning_score=15)
    (obs_p1, obs_p2), info = env.reset()
    
    # 렌더러 생성
    renderer = PygameRenderer(scale=2)
    
    done = False
    total_steps = 0
    
    try:
        while not done:
            # 종료 체크
            if renderer.check_quit():
                print("\n사용자가 종료했습니다.")
                break
            
            # 랜덤 행동
            action_p1 = env.action_space.sample()
            action_p2 = env.action_space.sample()
            
            # 환경 진행
            (obs_p1, obs_p2), (r_p1, r_p2), terminated, truncated, info = env.step((action_p1, action_p2))
            done = terminated or truncated
            total_steps += 1
            
            # 렌더링
            renderer.render(env.physics, info['score_p1'], info['score_p2'])
            
            # 득점 시 출력
            if r_p1 != 0:
                print(f"Score: {info['score_p1']} - {info['score_p2']}")
        
        if terminated:
            print(f"\n게임 종료! 최종 점수: {info['score_p1']} - {info['score_p2']}")
            winner = "Player 1" if info['score_p1'] > info['score_p2'] else "Player 2"
            print(f"승자: {winner}")
            
            # 잠시 결과 표시
            import time
            time.sleep(3)
    
    finally:
        renderer.close()
        env.close()
    
    print(f"\n총 {total_steps} 스텝 진행")
    print("데모 종료")


if __name__ == "__main__":
    main()

