"""
Pygame 렌더러 (원본 게임 그래픽 사용)

원본 view.js의 렌더링 방식을 정확히 따릅니다.
"""
import pygame
import json
import os
from typing import Dict, Tuple

from env.physics import PikachuPhysics


class PygameRenderer:
    """원본 게임 그래픽을 사용하는 렌더러"""
    
    def __init__(self, scale: int = 2):
        """
        Args:
            scale: 화면 확대 배율 (2배면 864x608 윈도우)
        """
        self.scale = scale
        self.width = 432 * scale
        self.height = 304 * scale
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pikachu Volleyball")
        self.clock = pygame.time.Clock()
        
        # 스프라이트 시트 로드
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets', 'images')
        sprite_sheet_path = os.path.join(assets_dir, 'sprite_sheet.png')
        sprite_json_path = os.path.join(assets_dir, 'sprite_sheet.json')
        
        self.sprite_sheet = pygame.image.load(sprite_sheet_path).convert_alpha()
        
        with open(sprite_json_path, 'r') as f:
            self.sprite_data = json.load(f)['frames']
        
        # 폰트
        self.font_large = pygame.font.Font(None, int(48 * scale))
        
        # 배경 스프라이트 미리 추출 (스케일 전)
        self.sprites = {}
        sprite_names = [
            "objects/sky_blue.png",
            "objects/mountain.png",
            "objects/ground_red.png",
            "objects/ground_line.png",
            "objects/ground_line_leftmost.png",
            "objects/ground_line_rightmost.png",
            "objects/ground_yellow.png",
            "objects/net_pillar_top.png",
            "objects/net_pillar.png",
            "objects/wave.png",
            "objects/shadow.png",
        ]
        for name in sprite_names:
            self.sprites[name] = self._get_sprite(name)[0]
    
    def _get_sprite(self, sprite_name: str) -> Tuple[pygame.Surface, Dict]:
        """스프라이트 추출 (스케일 적용)"""
        if sprite_name not in self.sprite_data:
            print(f"Warning: sprite '{sprite_name}' not found")
            # 기본 스프라이트
            sprite_name = list(self.sprite_data.keys())[0]
        
        frame_data = self.sprite_data[sprite_name]['frame']
        x, y, w, h = frame_data['x'], frame_data['y'], frame_data['w'], frame_data['h']
        
        sprite = pygame.Surface((w, h), pygame.SRCALPHA)
        sprite.blit(self.sprite_sheet, (0, 0), (x, y, w, h))
        
        # 스케일링
        if self.scale != 1:
            sprite = pygame.transform.scale(sprite, (w * self.scale, h * self.scale))
        
        return sprite, frame_data
    
    def render(self, physics: PikachuPhysics, score_p1: int, score_p2: int):
        """
        게임 화면 렌더링 (원본 view.js의 makeBGContainer 로직)
        
        Args:
            physics: 물리 엔진 객체
            score_p1: Player 1 점수
            score_p2: Player 2 점수
        """
        # 1. 하늘 배경 (16x16 타일, 12행 27열)
        sky_tile = self.sprites["objects/sky_blue.png"]
        for j in range(12):
            for i in range(27):  # 432 / 16 = 27
                self.screen.blit(sky_tile, (16 * i * self.scale, 16 * j * self.scale))
        
        # 2. 산 (y=188)
        mountain = self.sprites["objects/mountain.png"]
        self.screen.blit(mountain, (0, 188 * self.scale))
        
        # 3. ground_red (y=248, 16픽셀 타일)
        ground_red = self.sprites["objects/ground_red.png"]
        for i in range(27):
            self.screen.blit(ground_red, (16 * i * self.scale, 248 * self.scale))
        
        # 4. ground_line (y=264)
        ground_line = self.sprites["objects/ground_line.png"]
        for i in range(1, 26):  # 1부터 25까지 (양 끝 제외)
            self.screen.blit(ground_line, (16 * i * self.scale, 264 * self.scale))
        
        # 양 끝 라인
        ground_line_left = self.sprites["objects/ground_line_leftmost.png"]
        self.screen.blit(ground_line_left, (0, 264 * self.scale))
        
        ground_line_right = self.sprites["objects/ground_line_rightmost.png"]
        self.screen.blit(ground_line_right, (416 * self.scale, 264 * self.scale))  # 432 - 16
        
        # 5. ground_yellow (y=280, 296, 2줄)
        ground_yellow = self.sprites["objects/ground_yellow.png"]
        for j in range(2):
            for i in range(27):
                self.screen.blit(ground_yellow, (16 * i * self.scale, (280 + 16 * j) * self.scale))
        
        # 6. 네트 기둥
        net_pillar_top = self.sprites["objects/net_pillar_top.png"]
        self.screen.blit(net_pillar_top, (213 * self.scale, 176 * self.scale))
        
        net_pillar = self.sprites["objects/net_pillar.png"]
        for j in range(12):
            self.screen.blit(net_pillar, (213 * self.scale, (184 + 8 * j) * self.scale))
        
        # 7. 그림자
        shadow = self.sprites["objects/shadow.png"]
        shadow.set_alpha(128)  # 반투명
        
        # Player 1 그림자 (y=273)
        shadow_x_p1 = (physics.player1.x - shadow.get_width() // (2 * self.scale)) * self.scale
        self.screen.blit(shadow, (shadow_x_p1, 273 * self.scale))
        
        # Player 2 그림자
        shadow_x_p2 = (physics.player2.x - shadow.get_width() // (2 * self.scale)) * self.scale
        self.screen.blit(shadow, (shadow_x_p2, 273 * self.scale))
        
        # 공 그림자
        shadow_x_ball = (physics.ball.x - shadow.get_width() // (2 * self.scale)) * self.scale
        self.screen.blit(shadow, (shadow_x_ball, 273 * self.scale))
        
        # 8. Player 1 (좌측)
        p1_state = min(physics.player1.state, 6)
        p1_frame = physics.player1.frame_number % 5
        
        # state 3 (diving)은 프레임이 2개, state 4는 1개만
        if p1_state == 3:
            p1_frame = min(p1_frame, 1)
        elif p1_state == 4:
            p1_frame = 0
        
        p1_sprite_name = f"pikachu/pikachu_{p1_state}_{p1_frame}.png"
        p1_sprite, _ = self._get_sprite(p1_sprite_name)
        
        # 중심 좌표로 배치 (anchor 0.5, 0.5)
        p1_x = physics.player1.x * self.scale - p1_sprite.get_width() // 2
        p1_y = physics.player1.y * self.scale - p1_sprite.get_height() // 2
        self.screen.blit(p1_sprite, (p1_x, p1_y))
        
        # 9. Player 2 (우측) - 좌우 반전
        p2_state = min(physics.player2.state, 6)
        p2_frame = physics.player2.frame_number % 5
        
        if p2_state == 3:
            p2_frame = min(p2_frame, 1)
        elif p2_state == 4:
            p2_frame = 0
        
        p2_sprite_name = f"pikachu/pikachu_{p2_state}_{p2_frame}.png"
        p2_sprite, _ = self._get_sprite(p2_sprite_name)
        p2_sprite = pygame.transform.flip(p2_sprite, True, False)
        
        p2_x = physics.player2.x * self.scale - p2_sprite.get_width() // 2
        p2_y = physics.player2.y * self.scale - p2_sprite.get_height() // 2
        self.screen.blit(p2_sprite, (p2_x, p2_y))
        
        # 10. 공
        ball_rotation = physics.ball.rotation % 5
        ball_sprite_name = f"ball/ball_{ball_rotation}.png"
        ball_sprite, _ = self._get_sprite(ball_sprite_name)
        
        ball_x = physics.ball.x * self.scale - ball_sprite.get_width() // 2
        ball_y = physics.ball.y * self.scale - ball_sprite.get_height() // 2
        self.screen.blit(ball_sprite, (ball_x, ball_y))
        
        # 11. 파워히트 효과
        if physics.ball.punch_effect_radius > 0:
            punch_sprite, _ = self._get_sprite("ball/ball_punch.png")
            punch_sprite.set_alpha(180)
            
            punch_x = physics.ball.punch_effect_x * self.scale - punch_sprite.get_width() // 2
            punch_y = physics.ball.punch_effect_y * self.scale - punch_sprite.get_height() // 2
            self.screen.blit(punch_sprite, (punch_x, punch_y))
        
        # 12. 점수 표시
        score_text = self.font_large.render(f"{score_p1}  :  {score_p2}", True, (255, 255, 255))
        
        # 외곽선 효과
        score_outline = self.font_large.render(f"{score_p1}  :  {score_p2}", True, (0, 0, 0))
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            outline_rect = score_outline.get_rect(center=(self.width // 2 + dx, 30 * self.scale + dy))
            self.screen.blit(score_outline, outline_rect)
        
        score_rect = score_text.get_rect(center=(self.width // 2, 30 * self.scale))
        self.screen.blit(score_text, score_rect)
        
        # 화면 업데이트
        pygame.display.flip()
        self.clock.tick(25)  # 25 FPS
    
    def close(self):
        """렌더러 종료"""
        pygame.quit()
    
    def check_quit(self) -> bool:
        """종료 이벤트 체크"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False
