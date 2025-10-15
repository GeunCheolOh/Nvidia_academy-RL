"""
Gymnasium 환경 검증 스크립트

물리 엔진, 환경, 대칭성 등을 테스트합니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env import PikachuVolleyballEnv
from env.physics import PikachuPhysics, UserInput, PLAYER_TOUCHING_GROUND_Y
from env.symmetry import mirror_observation, mirror_action


def test_physics_basic():
    """기본 물리 엔진 테스트"""
    print("=" * 60)
    print("1. 기본 물리 엔진 테스트")
    print("=" * 60)
    
    physics = PikachuPhysics()
    
    print("\n초기 상태:")
    print(f"  Player 1: ({physics.player1.x:.1f}, {physics.player1.y:.1f})")
    print(f"  Player 2: ({physics.player2.x:.1f}, {physics.player2.y:.1f})")
    print(f"  Ball: ({physics.ball.x:.1f}, {physics.ball.y:.1f})")
    
    # 공이 떨어지는 시뮬레이션
    no_input = UserInput(0, 0, 0)
    for i in range(100):
        is_ground, scorer = physics.step(no_input, no_input)
        if is_ground:
            print(f"\n✓ {i}프레임 후 공이 바닥에 닿음")
            print(f"  Ball 위치: ({physics.ball.x:.1f}, {physics.ball.y:.1f})")
            print(f"  득점: Player {scorer}")
            break
    
    assert is_ground, "공이 바닥에 닿지 않았습니다"
    print("\n✓ 기본 물리 테스트 통과")


def test_player_movement():
    """플레이어 이동 테스트"""
    print("\n" + "=" * 60)
    print("2. 플레이어 이동 테스트")
    print("=" * 60)
    
    physics = PikachuPhysics()
    initial_x = physics.player1.x
    
    # 오른쪽으로 10프레임 이동
    move_right = UserInput(x_direction=1, y_direction=0, power_hit=0)
    no_input = UserInput(0, 0, 0)
    
    for _ in range(10):
        physics.step(move_right, no_input)
    
    moved_distance = physics.player1.x - initial_x
    print(f"\n  초기 위치: {initial_x:.1f}")
    print(f"  10프레임 후: {physics.player1.x:.1f}")
    print(f"  이동 거리: {moved_distance:.1f} (예상: 60)")
    
    assert moved_distance == 60, f"이동 거리가 잘못되었습니다: {moved_distance}"
    print("\n✓ 플레이어 이동 테스트 통과")


def test_jump():
    """점프 테스트"""
    print("\n" + "=" * 60)
    print("3. 점프 테스트")
    print("=" * 60)
    
    physics = PikachuPhysics()
    
    # 점프
    jump_input = UserInput(x_direction=0, y_direction=-1, power_hit=0)
    no_input = UserInput(0, 0, 0)
    
    max_height = physics.player1.y
    landed_frame = 0
    
    for i in range(50):
        physics.step(jump_input if i == 0 else no_input, no_input)
        if physics.player1.y < max_height:
            max_height = physics.player1.y
        if physics.player1.y == PLAYER_TOUCHING_GROUND_Y and i > 0:
            landed_frame = i
            break
    
    jump_height = PLAYER_TOUCHING_GROUND_Y - max_height
    print(f"\n  점프 후 {landed_frame}프레임 만에 착지")
    print(f"  최고 높이: {jump_height:.1f}픽셀")
    
    assert landed_frame > 0, "점프가 작동하지 않았습니다"
    assert jump_height > 100, f"점프 높이가 너무 낮습니다: {jump_height}"
    print("\n✓ 점프 테스트 통과")


def test_env_basic():
    """Gymnasium 환경 기본 테스트"""
    print("\n" + "=" * 60)
    print("4. Gymnasium 환경 기본 테스트")
    print("=" * 60)
    
    env = PikachuVolleyballEnv()
    
    # Reset
    (obs_p1, obs_p2), info = env.reset()
    print(f"\n✓ Reset 성공")
    print(f"  관찰 공간 shape: {obs_p1.shape}")
    print(f"  관찰 범위: [{obs_p1.min():.2f}, {obs_p1.max():.2f}]")
    print(f"  행동 공간: Discrete({env.action_space.n})")
    print(f"  초기 점수: {info['score_p1']}-{info['score_p2']}")
    
    # 관찰 벡터 검증
    assert obs_p1.shape == (15,), f"관찰 shape 오류: {obs_p1.shape}"
    assert obs_p2.shape == (15,), f"관찰 shape 오류: {obs_p2.shape}"
    assert obs_p1.dtype == np.float32, f"관찰 dtype 오류: {obs_p1.dtype}"
    
    # 랜덤 플레이
    done = False
    steps = 0
    scores_changed = 0
    
    print("\n랜덤 플레이 테스트 중...")
    while not done and steps < 1000:
        action_p1 = env.action_space.sample()
        action_p2 = env.action_space.sample()
        
        (obs_p1, obs_p2), (r_p1, r_p2), terminated, truncated, info = env.step((action_p1, action_p2))
        done = terminated or truncated
        steps += 1
        
        if r_p1 != 0 or r_p2 != 0:
            scores_changed += 1
            if scores_changed <= 3:  # 처음 3개만 출력
                print(f"  Step {steps}: 점수 {info['score_p1']}-{info['score_p2']}, 보상: ({r_p1:.1f}, {r_p2:.1f})")
    
    print(f"\n✓ {steps} 스텝 진행 완료")
    print(f"  최종 점수: {info['score_p1']}-{info['score_p2']}")
    print(f"  점수 변화 횟수: {scores_changed}")
    
    env.close()
    
    assert scores_changed > 0, "점수가 변하지 않았습니다"
    print("\n✓ 환경 기본 테스트 통과")


def test_symmetry():
    """대칭성 테스트"""
    print("\n" + "=" * 60)
    print("5. 좌우 대칭성 테스트")
    print("=" * 60)
    
    env = PikachuVolleyballEnv()
    (obs_p1, obs_p2), _ = env.reset()
    
    print("\nPlayer 1 관찰 (원본):")
    print(f"  P1 x: {obs_p1[0]:.3f}, P2 x: {obs_p1[4]:.3f}, Ball x: {obs_p1[8]:.3f}")
    print(f"  점수: P1={obs_p1[13]:.3f}, P2={obs_p1[14]:.3f}")
    
    print("\nPlayer 2 관찰 (반전됨):")
    print(f"  P1 x: {obs_p2[0]:.3f}, P2 x: {obs_p2[4]:.3f}, Ball x: {obs_p2[8]:.3f}")
    print(f"  점수: P1={obs_p2[13]:.3f}, P2={obs_p2[14]:.3f}")
    
    # 검증: obs_p2는 obs_p1을 반전한 것
    # Player 2가 보는 자신의 위치(obs_p2[0])는 반전된 Player 1의 Player 2 위치(1.0 - obs_p1[4])
    diff_p1_x = abs(obs_p2[0] - (1.0 - obs_p1[4]))
    diff_p2_x = abs(obs_p2[4] - (1.0 - obs_p1[0]))
    diff_ball_x = abs(obs_p2[8] - (1.0 - obs_p1[8]))
    
    print(f"\n대칭성 검증:")
    print(f"  P1 x 차이: {diff_p1_x:.6f}")
    print(f"  P2 x 차이: {diff_p2_x:.6f}")
    print(f"  Ball x 차이: {diff_ball_x:.6f}")
    
    assert diff_p1_x < 0.01, f"P1 x 좌표 대칭성 오류: {diff_p1_x}"
    assert diff_p2_x < 0.01, f"P2 x 좌표 대칭성 오류: {diff_p2_x}"
    assert diff_ball_x < 0.01, f"Ball x 좌표 대칭성 오류: {diff_ball_x}"
    
    # 행동 대칭성 테스트
    print("\n행동 대칭성 테스트:")
    action_tests = [
        (0, 0, "stay"),
        (1, 2, "left ↔ right"),
        (4, 5, "left+jump ↔ right+jump"),
        (7, 8, "left+dive ↔ right+dive"),
    ]
    
    for action, expected_mirror, desc in action_tests:
        mirrored = mirror_action(action)
        print(f"  {desc}: {action} → {mirrored}")
        assert mirrored == expected_mirror, f"행동 반전 오류: {action} → {mirrored} (예상: {expected_mirror})"
    
    env.close()
    print("\n✓ 대칭성 테스트 통과")


def test_full_game():
    """전체 게임 플레이 테스트"""
    print("\n" + "=" * 60)
    print("6. 전체 게임 테스트 (승리 조건)")
    print("=" * 60)
    
    # 빠른 테스트를 위해 3점제
    env = PikachuVolleyballEnv(winning_score=3)
    (obs_p1, obs_p2), info = env.reset()
    
    done = False
    total_steps = 0
    
    print("\n게임 진행 중...")
    while not done and total_steps < 5000:
        # 랜덤 행동
        action_p1 = env.action_space.sample()
        action_p2 = env.action_space.sample()
        
        (obs_p1, obs_p2), (r_p1, r_p2), terminated, truncated, info = env.step((action_p1, action_p2))
        done = terminated or truncated
        total_steps += 1
    
    print(f"\n게임 종료! ({total_steps} 스텝)")
    print(f"  최종 점수: {info['score_p1']}-{info['score_p2']}")
    
    winner = "Player 1" if info['score_p1'] > info['score_p2'] else "Player 2"
    print(f"  승자: {winner}")
    
    assert info['score_p1'] >= 3 or info['score_p2'] >= 3, "게임이 제대로 종료되지 않음"
    assert terminated, "게임이 terminated 되지 않음"
    
    env.close()
    print("\n✓ 전체 게임 테스트 통과")


def main():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("Pikachu Volleyball 환경 검증 테스트")
    print("="*60)
    
    try:
        test_physics_basic()
        test_player_movement()
        test_jump()
        test_env_basic()
        test_symmetry()
        test_full_game()
        
        print("\n" + "="*60)
        print("✅ 모든 테스트 통과!")
        print("="*60)
        print("\n환경이 올바르게 구현되었습니다.")
        print("다음 단계: Phase 2 (PPO 학습 시스템 구현)")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

