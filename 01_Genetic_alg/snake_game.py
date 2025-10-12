"""
Snake Game Core - 뱀 게임 핵심 로직
센서 기반 입력과 거리 기반 보상을 사용하는 개선된 버전
"""
import pygame
import random
import numpy as np

# 게임 설정
FPS = 15
SCREEN_SIZE = 30
PIXEL_SIZE = 20
LINE_WIDTH = 1

# 방향 벡터: 상, 우, 하, 좌
DIRECTIONS = np.array([
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (0, 1),   # DOWN
    (-1, 0)   # LEFT
])


class SnakeGame:
    """뱀 게임 클래스 - 센서 기반 입력 사용"""
    
    def __init__(self, screen, genome=None):
        """
        초기화
        
        Args:
            screen: Pygame 화면 객체
            genome: 유전 알고리즘 유전자 (None이면 수동 플레이)
        """
        self.genome = genome
        self.screen = screen
        self.score = 0
        self.snake = np.array([[15, 26], [15, 27], [15, 28], [15, 29]])  # 초기 뱀 위치
        self.direction = 0  # 초기 방향 (위쪽)
        self.place_fruit()
        self.timer = 0
        self.last_fruit_time = 0
        
        # Fitness 관련
        self.fitness = 0.0
        self.last_dist = np.inf
    
    def place_fruit(self):
        """과일을 랜덤 위치에 배치"""
        while True:
            x = random.randint(0, SCREEN_SIZE - 1)
            y = random.randint(0, SCREEN_SIZE - 1)
            if [x, y] not in self.snake.tolist():
                break
        self.fruit = np.array([x, y])
    
    def step(self, direction):
        """
        한 스텝 진행
        
        Args:
            direction: 이동할 방향 (0~3)
            
        Returns:
            성공 여부 (충돌하면 False)
        """
        old_head = self.snake[0]
        movement = DIRECTIONS[direction]
        new_head = old_head + movement
        
        # 벽 또는 자기 자신과 충돌 체크
        if (new_head[0] < 0 or new_head[0] >= SCREEN_SIZE or
            new_head[1] < 0 or new_head[1] >= SCREEN_SIZE or
            new_head.tolist() in self.snake.tolist()):
            return False
        
        # 과일 먹기
        if np.array_equal(new_head, self.fruit):
            self.last_fruit_time = self.timer
            self.score += 1
            self.fitness += 10  # 과일 먹으면 큰 보상
            self.place_fruit()
        else:
            # 꼬리 제거
            self.snake = self.snake[:-1, :]
        
        # 머리 추가
        self.snake = np.concatenate([[new_head], self.snake], axis=0)
        return True
    
    def get_sensor_inputs(self):
        """
        센서 기반 입력 생성 (6개 값)
        - 직진/좌/우 방향의 장애물 거리 (0~1, 0=위험)
        - 과일이 직진/좌/우 중 어느 방향에 있는지 (원-핫)
        
        Returns:
            6차원 입력 벡터
        """
        head = self.snake[0]
        result = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        
        # 3방향 확인: 직진, 좌회전, 우회전
        possible_dirs = [
            DIRECTIONS[self.direction],
            DIRECTIONS[(self.direction + 3) % 4],  # 좌회전
            DIRECTIONS[(self.direction + 1) % 4]   # 우회전
        ]
        
        # 장애물 감지 (5칸 앞까지)
        for i, p_dir in enumerate(possible_dirs):
            for j in range(5):
                check_pos = head + p_dir * (j + 1)
                if (check_pos[0] < 0 or check_pos[0] >= SCREEN_SIZE or
                    check_pos[1] < 0 or check_pos[1] >= SCREEN_SIZE or
                    check_pos.tolist() in self.snake.tolist()):
                    result[i] = j * 0.2  # 거리에 따라 0~1 값
                    break
        
        # 과일 방향 감지
        # 직진 방향에 과일이 있는지
        if (np.any(head == self.fruit) and 
            np.sum(head * possible_dirs[0]) <= np.sum(self.fruit * possible_dirs[0])):
            result[3] = 1
        # 좌측에 과일이 있는지
        if np.sum(head * possible_dirs[1]) < np.sum(self.fruit * possible_dirs[1]):
            result[4] = 1
        # 우측에 과일이 있는지
        else:
            result[5] = 1
        
        return np.array(result)
    
    def render(self):
        """화면 렌더링"""
        # 배경
        self.screen.fill((0, 0, 0))
        
        # 벽
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        [0, 0, SCREEN_SIZE*PIXEL_SIZE, LINE_WIDTH])
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        [0, SCREEN_SIZE*PIXEL_SIZE-LINE_WIDTH, 
                         SCREEN_SIZE*PIXEL_SIZE, LINE_WIDTH])
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        [0, 0, LINE_WIDTH, SCREEN_SIZE*PIXEL_SIZE])
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        [SCREEN_SIZE*PIXEL_SIZE-LINE_WIDTH, 0, 
                         LINE_WIDTH, SCREEN_SIZE*PIXEL_SIZE+LINE_WIDTH])
        
        # 뱀
        for segment in self.snake:
            rect = pygame.Rect(segment[0] * PIXEL_SIZE, segment[1] * PIXEL_SIZE,
                             PIXEL_SIZE, PIXEL_SIZE)
            pygame.draw.rect(self.screen, (255, 0, 0), rect)
        
        # 과일
        fruit_rect = pygame.Rect(self.fruit[0] * PIXEL_SIZE, self.fruit[1] * PIXEL_SIZE,
                                PIXEL_SIZE, PIXEL_SIZE)
        pygame.draw.rect(self.screen, (0, 255, 0), fruit_rect)
        
        # 점수
        font = pygame.font.Font(None, 20)
        font.set_bold(True)
        score_text = font.render(str(self.score), False, (255, 255, 255))
        self.screen.blit(score_text, (5, 5))
        
        pygame.display.update()
    
    def run(self, manual=False, training_mode=False, render=True):
        """
        게임 실행
        
        Args:
            manual: True면 키보드로 수동 조작
            training_mode: True면 학습 모드 (시간 제한/fitness 페널티 적용)
            render: True면 화면 렌더링 (False시 학습 속도 향상)
            
        Returns:
            (fitness, score) 튜플
        """
        self.fitness = 0
        clock = pygame.time.Clock()
        prev_key = pygame.K_UP
        
        while True:
            self.timer += 0.1
            
            # 종료 조건 (학습 모드에서만 적용)
            # fitness가 너무 낮거나 5초간 과일을 못 먹으면 종료
            if training_mode:
                if self.fitness < -FPS/2 or self.timer - self.last_fruit_time > 0.1 * FPS * 5:
                    break
            
            # FPS 제한 (렌더링 모드에서만)
            if render:
                clock.tick(FPS)
            
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self.fitness, self.score
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()
                    if event.key == pygame.K_SPACE:
                        # 일시정지
                        pause = True
                        while pause:
                            for e in pygame.event.get():
                                if e.type == pygame.QUIT:
                                    pygame.quit()
                                    return self.fitness, self.score
                                elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                    pause = False
                    
                    # 수동 플레이 키 입력
                    if manual:
                        if prev_key != pygame.K_DOWN and event.key == pygame.K_UP:
                            self.direction = 0
                            prev_key = event.key
                        elif prev_key != pygame.K_LEFT and event.key == pygame.K_RIGHT:
                            self.direction = 1
                            prev_key = event.key
                        elif prev_key != pygame.K_UP and event.key == pygame.K_DOWN:
                            self.direction = 2
                            prev_key = event.key
                        elif prev_key != pygame.K_RIGHT and event.key == pygame.K_LEFT:
                            self.direction = 3
                            prev_key = event.key
            
            # AI 플레이
            if not manual and self.genome is not None:
                inputs = self.get_sensor_inputs()
                outputs = self.genome.forward(inputs)
                action = np.argmax(outputs)
                
                if action == 0:  # 직진
                    pass
                elif action == 1:  # 좌회전
                    self.direction = (self.direction + 3) % 4
                elif action == 2:  # 우회전
                    self.direction = (self.direction + 1) % 4
            
            # 이동
            if not self.step(self.direction):
                break
            
            # Fitness 계산 (학습 모드에서만 거리 기반 보상 적용)
            if training_mode:
                current_dist = np.linalg.norm(self.snake[0] - self.fruit)
                if self.last_dist > current_dist:
                    self.fitness += 1.0  # 가까워지면 보상
                else:
                    self.fitness -= 1.5  # 멀어지면 페널티
                self.last_dist = current_dist
            
            # 렌더링 (render=True일 때만)
            if render:
                self.render()
        
        return self.fitness, self.score

