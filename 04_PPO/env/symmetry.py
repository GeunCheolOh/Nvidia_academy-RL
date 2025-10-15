"""
좌우 대칭 변환 유틸리티

Self-play를 위해 Player 2의 관찰을 Player 1 시점으로 변환합니다.
"""
import numpy as np
from typing import Tuple


def mirror_observation(obs: np.ndarray) -> np.ndarray:
    """
    Player 2의 관찰을 Player 1 시점으로 변환
    
    관찰 벡터 구조 (15차원):
    [0-3]   player1_x, player1_y, player1_y_vel, player1_state,
    [4-7]   player2_x, player2_y, player2_y_vel, player2_state,
    [8-12]  ball_x, ball_y, ball_x_vel, ball_y_vel, ball_expected_x,
    [13-14] my_score, opponent_score
    
    Args:
        obs: 원본 관찰 (Player 2 시점)
    
    Returns:
        반전된 관찰 (Player 1 시점으로 변환)
    """
    mirrored = obs.copy()
    
    # 플레이어 정보 교환 (player1 ↔ player2)
    mirrored[0:4] = obs[4:8]  # player2 → player1 위치에
    mirrored[4:8] = obs[0:4]  # player1 → player2 위치에
    
    # X 좌표 반전 (정규화되어 있다고 가정: 0~1)
    mirrored[0] = 1.0 - mirrored[0]  # player1_x
    mirrored[4] = 1.0 - mirrored[4]  # player2_x
    mirrored[8] = 1.0 - mirrored[8]  # ball_x
    mirrored[12] = 1.0 - mirrored[12]  # ball_expected_x
    
    # X 방향 속도 반전
    mirrored[10] = -mirrored[10]  # ball_x_vel
    
    # 점수 교환
    mirrored[13] = obs[14]  # opponent_score → my_score
    mirrored[14] = obs[13]  # my_score → opponent_score
    
    return mirrored


def mirror_action(action: int) -> int:
    """
    행동 좌우 반전 (Discrete action space용 - 레거시)
    
    행동 매핑 (9가지):
    0: stay          → 0: stay
    1: left          → 2: right
    2: right         → 1: left
    3: jump          → 3: jump
    4: left+jump     → 5: right+jump
    5: right+jump    → 4: left+jump
    6: power_hit     → 6: power_hit
    7: left+dive     → 8: right+dive
    8: right+dive    → 7: left+dive
    
    Args:
        action: 원본 행동
    
    Returns:
        반전된 행동
    """
    action_mirror_map = {
        0: 0,  # stay
        1: 2,  # left ↔ right
        2: 1,  # right ↔ left
        3: 3,  # jump
        4: 5,  # left+jump ↔ right+jump
        5: 4,  # right+jump ↔ left+jump
        6: 6,  # power_hit
        7: 8,  # left+dive ↔ right+dive
        8: 7,  # right+dive ↔ left+dive
    }
    return action_mirror_map.get(action, action)


def mirror_action_multidiscrete(action: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    행동 좌우 반전 (Multi-Discrete action space용)
    
    Action = (x_dir, y_dir, power_hit)
    - x_dir: 0(left), 1(stay), 2(right) → 반전
    - y_dir: 0(up), 1(stay), 2(down) → 그대로
    - power_hit: 0(no), 1(yes) → 그대로
    
    Args:
        action: (x_dir_idx, y_dir_idx, power_hit_idx)
    
    Returns:
        반전된 행동
    """
    x_dir, y_dir, power_hit = action
    
    # x_direction만 반전: 0 ↔ 2, 1은 그대로
    x_dir_mirror_map = {0: 2, 1: 1, 2: 0}
    
    return (x_dir_mirror_map[x_dir], y_dir, power_hit)
