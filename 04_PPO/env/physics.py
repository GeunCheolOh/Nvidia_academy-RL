"""
Pikachu Volleyball Physics Engine (Python port of physics.js)

이 파일은 원본 게임의 physics.js를 Python으로 포팅한 것입니다.
"""
import numpy as np
from typing import Tuple, NamedTuple

# 물리 상수
GROUND_WIDTH = 432
GROUND_HEIGHT = 304
GROUND_HALF_WIDTH = 216

PLAYER_LENGTH = 64
PLAYER_HALF_LENGTH = 32
PLAYER_TOUCHING_GROUND_Y = 244

BALL_RADIUS = 20
BALL_TOUCHING_GROUND_Y = 252

NET_PILLAR_HALF_WIDTH = 25
NET_PILLAR_TOP_TOP_Y = 176
NET_PILLAR_TOP_BOTTOM_Y = 192

FPS = 25
INFINITE_LOOP_LIMIT = 1000


class UserInput(NamedTuple):
    """플레이어 입력"""
    x_direction: int  # -1: 좌, 0: 정지, 1: 우
    y_direction: int  # -1: 위, 0: 정지, 1: 아래
    power_hit: int    # 0: 없음, 1: 파워히트


class Player:
    """플레이어 클래스"""
    
    def __init__(self, is_player2: bool, is_computer: bool = False):
        self.is_player2 = is_player2
        self.is_computer = is_computer
        
        # 라운드마다 초기화되는 속성
        self.x = 0
        self.y = 0
        self.y_velocity = 0
        self.is_collision_with_ball_happened = False
        self.state = 0
        self.frame_number = 0
        self.normal_status_arm_swing_direction = 1
        self.delay_before_next_frame = 0
        
        # 지속적인 속성
        self.diving_direction = 0
        self.lying_down_duration_left = -1
        self.is_winner = False
        self.game_ended = False
        self.computer_boldness = np.random.randint(0, 5)
        self.computer_where_to_stand_by = 0
        
        self.initialize_for_new_round()
    
    def initialize_for_new_round(self):
        """라운드 시작 시 초기화"""
        self.x = 36 if not self.is_player2 else GROUND_WIDTH - 36
        self.y = PLAYER_TOUCHING_GROUND_Y
        self.y_velocity = 0
        self.is_collision_with_ball_happened = False
        self.state = 0  # 0: normal, 1: jumping, 2: power_hitting, 3: diving, 4: lying_down, 5: win, 6: lost
        self.frame_number = 0
        self.normal_status_arm_swing_direction = 1
        self.delay_before_next_frame = 0


class Ball:
    """공 클래스"""
    
    def __init__(self, is_player2_serve: bool = False):
        # 라운드마다 초기화되는 속성
        self.x = 0
        self.y = 0
        self.x_velocity = 0
        self.y_velocity = 0
        self.is_power_hit = False
        
        # 지속적인 속성
        self.expected_landing_point_x = 0
        self.rotation = 0
        self.fine_rotation = 0
        self.punch_effect_x = 0
        self.punch_effect_y = 0
        self.punch_effect_radius = 0
        self.previous_x = 0
        self.previous_previous_x = 0
        self.previous_y = 0
        self.previous_previous_y = 0
        
        self.initialize_for_new_round(is_player2_serve)
    
    def initialize_for_new_round(self, is_player2_serve: bool):
        """라운드 시작 시 초기화"""
        self.x = 56 if not is_player2_serve else GROUND_WIDTH - 56
        self.y = 0
        self.x_velocity = 0
        self.y_velocity = 1
        self.is_power_hit = False


class PikachuPhysics:
    """Pikachu Volleyball 물리 엔진"""
    
    def __init__(self):
        self.player1 = Player(is_player2=False)
        self.player2 = Player(is_player2=True)
        self.ball = Ball(is_player2_serve=False)
    
    def step(self, user_input_p1: UserInput, user_input_p2: UserInput) -> Tuple[bool, int]:
        """
        한 프레임 진행
        
        Returns:
            is_ball_touching_ground: 공이 바닥에 닿았는지
            scoring_player: 득점한 플레이어 (0: 없음, 1: player1, 2: player2)
        """
        # 예상 착지점 계산
        self._calculate_expected_landing_point()
        
        # 플레이어 물리 처리
        self._process_player_physics(self.player1, user_input_p1, self.player2)
        self._process_player_physics(self.player2, user_input_p2, self.player1)
        
        # 충돌 처리
        self._process_collision(self.player1, user_input_p1)
        self._process_collision(self.player2, user_input_p2)
        
        # 공 물리 처리
        is_ball_touching_ground = self._process_ball_physics()
        
        # 득점 판정
        scoring_player = 0
        if is_ball_touching_ground:
            if self.ball.punch_effect_x < GROUND_HALF_WIDTH:
                scoring_player = 2  # Player 2 득점
            else:
                scoring_player = 1  # Player 1 득점
        
        return is_ball_touching_ground, scoring_player
    
    def _process_ball_physics(self) -> bool:
        """공 물리 처리"""
        # 이전 위치 저장
        self.ball.previous_previous_x = self.ball.previous_x
        self.ball.previous_previous_y = self.ball.previous_y
        self.ball.previous_x = self.ball.x
        self.ball.previous_y = self.ball.y
        
        # 회전 처리
        future_fine_rotation = self.ball.fine_rotation + (self.ball.x_velocity // 2)
        if future_fine_rotation < 0:
            future_fine_rotation += 50
        elif future_fine_rotation > 50:
            future_fine_rotation -= 50
        self.ball.fine_rotation = future_fine_rotation
        self.ball.rotation = future_fine_rotation // 10
        
        # 좌우 벽 반사
        future_ball_x = self.ball.x + self.ball.x_velocity
        if future_ball_x < BALL_RADIUS or future_ball_x > GROUND_WIDTH:
            self.ball.x_velocity = -self.ball.x_velocity
        
        # 천장 반사
        future_ball_y = self.ball.y + self.ball.y_velocity
        if future_ball_y < 0:
            self.ball.y_velocity = 1
        
        # 네트 충돌
        if abs(self.ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and \
           self.ball.y > NET_PILLAR_TOP_TOP_Y:
            if self.ball.y <= NET_PILLAR_TOP_BOTTOM_Y:
                if self.ball.y_velocity > 0:
                    self.ball.y_velocity = -self.ball.y_velocity
            else:
                if self.ball.x < GROUND_HALF_WIDTH:
                    self.ball.x_velocity = -abs(self.ball.x_velocity)
                else:
                    self.ball.x_velocity = abs(self.ball.x_velocity)
        
        # 바닥 충돌
        future_ball_y = self.ball.y + self.ball.y_velocity
        if future_ball_y > BALL_TOUCHING_GROUND_Y:
            self.ball.y_velocity = -self.ball.y_velocity
            self.ball.punch_effect_x = self.ball.x
            self.ball.y = BALL_TOUCHING_GROUND_Y
            self.ball.punch_effect_radius = BALL_RADIUS
            self.ball.punch_effect_y = BALL_TOUCHING_GROUND_Y + BALL_RADIUS
            return True
        
        # 위치 업데이트
        self.ball.y = future_ball_y
        self.ball.x = self.ball.x + self.ball.x_velocity
        self.ball.y_velocity += 1  # 중력
        
        return False
    
    def _process_player_physics(self, player: Player, user_input: UserInput, other_player: Player):
        """플레이어 물리 처리"""
        # 누워있는 상태 처리
        if player.state == 4:
            player.lying_down_duration_left -= 1
            if player.lying_down_duration_left < -1:
                player.state = 0
            return
        
        # X 방향 이동
        player_velocity_x = 0
        if player.state < 5:
            if player.state < 3:
                player_velocity_x = user_input.x_direction * 6
            else:  # diving
                player_velocity_x = player.diving_direction * 8
        
        player.x += player_velocity_x
        
        # 경계 처리
        if not player.is_player2:
            player.x = np.clip(player.x, PLAYER_HALF_LENGTH, GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH)
        else:
            player.x = np.clip(player.x, GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH, GROUND_WIDTH - PLAYER_HALF_LENGTH)
        
        # 점프
        if player.state < 3 and user_input.y_direction == -1 and player.y == PLAYER_TOUCHING_GROUND_Y:
            player.y_velocity = -16
            player.state = 1
            player.frame_number = 0
        
        # Y 방향 이동 (중력)
        player.y += player.y_velocity
        if player.y < PLAYER_TOUCHING_GROUND_Y:
            player.y_velocity += 1
        elif player.y > PLAYER_TOUCHING_GROUND_Y:
            player.y_velocity = 0
            player.y = PLAYER_TOUCHING_GROUND_Y
            player.frame_number = 0
            if player.state == 3:
                player.state = 4
                player.lying_down_duration_left = 3
            else:
                player.state = 0
        
        # 파워히트 / 다이빙
        if user_input.power_hit == 1:
            if player.state == 1:  # 점프 중 → 파워히트
                player.delay_before_next_frame = 5
                player.frame_number = 0
                player.state = 2
            elif player.state == 0 and user_input.x_direction != 0:  # 바닥에서 → 다이빙
                player.state = 3
                player.frame_number = 0
                player.diving_direction = user_input.x_direction
                player.y_velocity = -5
        
        # 프레임 업데이트
        if player.state == 1:
            player.frame_number = (player.frame_number + 1) % 3
        elif player.state == 2:
            if player.delay_before_next_frame < 1:
                player.frame_number += 1
                if player.frame_number > 4:
                    player.frame_number = 0
                    player.state = 1
            else:
                player.delay_before_next_frame -= 1
        elif player.state == 0:
            player.delay_before_next_frame += 1
            if player.delay_before_next_frame > 3:
                player.delay_before_next_frame = 0
                future_frame_number = player.frame_number + player.normal_status_arm_swing_direction
                if future_frame_number < 0 or future_frame_number > 4:
                    player.normal_status_arm_swing_direction = -player.normal_status_arm_swing_direction
                player.frame_number = player.frame_number + player.normal_status_arm_swing_direction
    
    def _process_collision(self, player: Player, user_input: UserInput):
        """공과 플레이어 충돌 처리"""
        # 충돌 판정
        is_collision = abs(self.ball.x - player.x) <= PLAYER_HALF_LENGTH and \
                       abs(self.ball.y - player.y) <= PLAYER_HALF_LENGTH
        
        if is_collision:
            if not player.is_collision_with_ball_happened:
                # 공 속도 변경
                if self.ball.x < player.x:
                    self.ball.x_velocity = -(abs(self.ball.x - player.x) // 3)
                elif self.ball.x > player.x:
                    self.ball.x_velocity = abs(self.ball.x - player.x) // 3
                
                if self.ball.x_velocity == 0:
                    self.ball.x_velocity = np.random.randint(-1, 2)
                
                ball_abs_y_velocity = abs(self.ball.y_velocity)
                self.ball.y_velocity = -ball_abs_y_velocity
                if ball_abs_y_velocity < 15:
                    self.ball.y_velocity = -15
                
                # 파워히트
                if player.state == 2:
                    if self.ball.x < GROUND_HALF_WIDTH:
                        self.ball.x_velocity = (abs(user_input.x_direction) + 1) * 10
                    else:
                        self.ball.x_velocity = -(abs(user_input.x_direction) + 1) * 10
                    
                    self.ball.punch_effect_x = self.ball.x
                    self.ball.punch_effect_y = self.ball.y
                    self.ball.y_velocity = abs(self.ball.y_velocity) * user_input.y_direction * 2
                    self.ball.punch_effect_radius = BALL_RADIUS
                    self.ball.is_power_hit = True
                else:
                    self.ball.is_power_hit = False
                
                # 예상 착지점 계산
                self._calculate_expected_landing_point()
                
                player.is_collision_with_ball_happened = True
        else:
            player.is_collision_with_ball_happened = False
    
    def _calculate_expected_landing_point(self):
        """공의 예상 착지점 계산 (시뮬레이션)"""
        # 공 상태 복사
        copy_x = float(self.ball.x)
        copy_y = float(self.ball.y)
        copy_x_vel = float(self.ball.x_velocity)
        copy_y_vel = float(self.ball.y_velocity)
        
        loop_count = 0
        
        while loop_count < INFINITE_LOOP_LIMIT:
            loop_count += 1
            
            # 벽 반사
            future_x = copy_x + copy_x_vel
            if future_x < BALL_RADIUS or future_x > GROUND_WIDTH:
                copy_x_vel = -copy_x_vel
            
            # 천장 반사
            if copy_y + copy_y_vel < 0:
                copy_y_vel = 1
            
            # 네트 충돌
            if abs(copy_x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_y > NET_PILLAR_TOP_TOP_Y:
                if copy_y < NET_PILLAR_TOP_BOTTOM_Y:
                    if copy_y_vel > 0:
                        copy_y_vel = -copy_y_vel
                else:
                    if copy_x < GROUND_HALF_WIDTH:
                        copy_x_vel = -abs(copy_x_vel)
                    else:
                        copy_x_vel = abs(copy_x_vel)
            
            copy_y += copy_y_vel
            
            # 바닥 도달
            if copy_y > BALL_TOUCHING_GROUND_Y:
                break
            
            copy_x += copy_x_vel
            copy_y_vel += 1
        
        self.ball.expected_landing_point_x = int(copy_x)

