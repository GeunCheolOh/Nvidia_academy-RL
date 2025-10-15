"""
Pikachu Volleyball Gymnasium Environment (Multi-Discrete Action Space)

Action = (x_direction, y_direction, power_hit)
- x_direction: {-1, 0, 1} = {0, 1, 2}
- y_direction: {-1, 0, 1} = {0, 1, 2}
- power_hit: {0, 1}

총 3 * 3 * 2 = 18가지 조합
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from .physics import PikachuPhysics, UserInput, GROUND_WIDTH, GROUND_HEIGHT
from .symmetry import mirror_observation


class PikachuVolleyballEnvMultiDiscrete(gym.Env):
    """
    Multi-Discrete Action Space를 사용하는 Pikachu Volleyball 환경
    
    Self-play를 위해 두 에이전트가 동시에 행동하는 환경입니다.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}
    
    def __init__(self, render_mode: Optional[str] = None, winning_score: int = 15):
        super().__init__()
        
        self.winning_score = winning_score
        self.render_mode = render_mode
        
        # 행동 공간: MultiDiscrete [3, 3, 2]
        # [x_direction(3), y_direction(3), power_hit(2)]
        self.action_space = spaces.MultiDiscrete([3, 3, 2])
        
        # 관찰 공간: 연속값 15차원
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )
        
        # 물리 엔진
        self.physics = None
        
        # 게임 상태
        self.score_p1 = 0
        self.score_p2 = 0
        self.total_frames = 0
        self.round_frames = 0
        self.is_player2_serve = False
        
        # 렌더링 (선택적)
        self.window = None
        self.clock = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        """환경 리셋"""
        super().reset(seed=seed)
        
        # 물리 엔진 초기화
        self.physics = PikachuPhysics()
        
        # 점수 초기화
        self.score_p1 = 0
        self.score_p2 = 0
        self.total_frames = 0
        self.round_frames = 0
        self.is_player2_serve = False
        
        obs_p1 = self._get_observation(player=1)
        obs_p2 = self._get_observation(player=2)
        
        info = {
            "score_p1": self.score_p1,
            "score_p2": self.score_p2,
            "round": 1,
        }
        
        return (obs_p1, obs_p2), info
    
    def step(
        self, actions: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, bool, Dict[str, Any]]:
        """환경 진행"""
        action_p1, action_p2 = actions
        
        # 행동을 UserInput으로 변환
        input_p1 = self._action_to_user_input(action_p1)
        input_p2 = self._action_to_user_input(action_p2)
        
        # 물리 엔진 진행
        is_ball_touching_ground, scoring_player = self.physics.step(input_p1, input_p2)
        
        # 프레임 카운터
        self.total_frames += 1
        self.round_frames += 1
        
        # 보상 계산
        reward_p1, reward_p2 = 0.0, 0.0
        terminated = False
        
        if is_ball_touching_ground:
            # 득점 처리
            if scoring_player == 1:
                self.score_p1 += 1
                reward_p1 = 1.0
                reward_p2 = -1.0
                self.is_player2_serve = False
            elif scoring_player == 2:
                self.score_p2 += 1
                reward_p1 = -1.0
                reward_p2 = 1.0
                self.is_player2_serve = True
            
            # 승리 판정
            if self.score_p1 >= self.winning_score:
                reward_p1 += 10.0
                reward_p2 -= 10.0
                terminated = True
            elif self.score_p2 >= self.winning_score:
                reward_p1 -= 10.0
                reward_p2 += 10.0
                terminated = True
            
            # 라운드 종료 - 새 라운드 시작
            if not terminated:
                self.physics.player1.initialize_for_new_round()
                self.physics.player2.initialize_for_new_round()
                self.physics.ball.initialize_for_new_round(self.is_player2_serve)
                self.round_frames = 0
        
        # 관찰 획득
        obs_p1 = self._get_observation(player=1)
        obs_p2 = self._get_observation(player=2)
        
        # Truncation
        truncated = self.total_frames >= 10000
        
        info = {
            "score_p1": self.score_p1,
            "score_p2": self.score_p2,
            "total_frames": self.total_frames,
            "round_frames": self.round_frames,
            "is_ball_touching_ground": is_ball_touching_ground,
        }
        
        if self.render_mode == "human":
            self.render()
        
        return (obs_p1, obs_p2), (reward_p1, reward_p2), terminated, truncated, info
    
    def _get_observation(self, player: int) -> np.ndarray:
        """관찰 획득 (플레이어 시점)"""
        # 정규화된 관찰 생성
        obs = np.array([
            # Player 1
            self.physics.player1.x / GROUND_WIDTH,
            self.physics.player1.y / GROUND_HEIGHT,
            self.physics.player1.y_velocity / 20.0,
            self.physics.player1.state / 6.0,
            
            # Player 2
            self.physics.player2.x / GROUND_WIDTH,
            self.physics.player2.y / GROUND_HEIGHT,
            self.physics.player2.y_velocity / 20.0,
            self.physics.player2.state / 6.0,
            
            # Ball
            self.physics.ball.x / GROUND_WIDTH,
            self.physics.ball.y / GROUND_HEIGHT,
            self.physics.ball.x_velocity / 20.0,
            self.physics.ball.y_velocity / 20.0,
            self.physics.ball.expected_landing_point_x / GROUND_WIDTH,
            
            # Scores
            self.score_p1 / float(self.winning_score),
            self.score_p2 / float(self.winning_score),
        ], dtype=np.float32)
        
        # Player 2는 좌우 반전
        if player == 2:
            obs = mirror_observation(obs)
        
        return obs
    
    def _action_to_user_input(self, action: np.ndarray) -> UserInput:
        """
        MultiDiscrete 행동을 UserInput으로 변환
        
        Args:
            action: [x_dir_idx, y_dir_idx, power_hit]
                - x_dir_idx: 0, 1, 2 → -1, 0, 1
                - y_dir_idx: 0, 1, 2 → -1, 0, 1
                - power_hit: 0, 1
        """
        x_dir = int(action[0]) - 1  # 0,1,2 → -1,0,1
        y_dir = int(action[1]) - 1  # 0,1,2 → -1,0,1
        power_hit = int(action[2])  # 0,1
        
        return UserInput(x_dir, y_dir, power_hit)
    
    def render(self):
        """렌더링"""
        if self.render_mode is None:
            return
        pass
    
    def close(self):
        """환경 종료"""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

