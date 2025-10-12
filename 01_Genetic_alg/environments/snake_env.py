"""
Snake Game Environment with Pygame GUI Support
"""
import numpy as np
import random
from typing import Tuple, List, Optional, Dict
from core.base import Environment

# Try to import pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
    print("[OK] Pygame is available")
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] Pygame not available, using text-based visualization")


class SnakeEnvironment(Environment):
    """Snake Game Environment with GUI Support"""
    
    def __init__(self, width: int = 20, height: int = 20, render_mode: bool = False):
        """
        Args:
            width: Game board width
            height: Game board height
            render_mode: Whether to render the game with Pygame
        """
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # Game state
        self.snake = []
        self.food = None
        self.direction = 0  # 0: Up, 1: Right, 2: Down, 3: Left
        self.score = 0
        self.steps = 0
        self.max_steps = width * height * 2  # Prevent infinite games
        
        # Action space: 4 directions
        self.action_size = 4
        
        # State representation
        self.state_shape = (width, height, 3)  # 3 channels: snake body, snake head, food
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.cell_size = 20
        
        if render_mode and PYGAME_AVAILABLE:
            self._init_pygame()
    
    def _init_pygame(self):
        """Initialize Pygame"""
        pygame.init()
        window_width = self.width * self.cell_size
        window_height = self.height * self.cell_size + 60  # Extra space for score
        
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Snake Game - Genetic Algorithm")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'background': (0, 0, 0),
            'snake_head': (0, 255, 0),
            'snake_body': (0, 200, 0),
            'food': (255, 0, 0),
            'text': (255, 255, 255)
        }
        
        # Font
        self.font = pygame.font.Font(None, 36)
    
    def reset(self) -> np.ndarray:
        """Reset the game"""
        # Initialize snake in the center
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = [(center_x, center_y)]
        self.direction = 0  # Start moving up
        self.score = 0
        self.steps = 0
        
        # Place initial food
        self._place_food()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step"""
        self.steps += 1
        
        # Update direction (prevent 180-degree turns)
        if action == 0 and self.direction != 2:  # Up
            self.direction = 0
        elif action == 1 and self.direction != 3:  # Right
            self.direction = 1
        elif action == 2 and self.direction != 0:  # Down
            self.direction = 2
        elif action == 3 and self.direction != 1:  # Left
            self.direction = 3
        
        # Move snake
        head_x, head_y = self.snake[0]
        if self.direction == 0:  # Up
            new_head = (head_x, head_y - 1)
        elif self.direction == 1:  # Right
            new_head = (head_x + 1, head_y)
        elif self.direction == 2:  # Down
            new_head = (head_x, head_y + 1)
        elif self.direction == 3:  # Left
            new_head = (head_x - 1, head_y)
        
        # Check collisions
        reward = 0
        done = False
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            reward = -10
            done = True
        
        # Self collision
        elif new_head in self.snake:
            reward = -10
            done = True
        
        # Food collision
        elif new_head == self.food:
            self.snake.insert(0, new_head)
            self.score += 1
            reward = 10
            self._place_food()
        
        # Normal move
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = -0.01  # Small penalty to encourage efficiency
        
        # Check maximum steps
        if self.steps >= self.max_steps:
            done = True
        
        # Render if needed
        if self.render_mode and self.screen is not None:
            self.render()
        
        return self._get_state(), reward, done, {'score': self.score, 'steps': self.steps}
    
    def _place_food(self):
        """Place food randomly on the board"""
        available_positions = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.snake:
                    available_positions.append((x, y))
        
        if available_positions:
            self.food = random.choice(available_positions)
        else:
            # Game won (all cells filled)
            self.food = None
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = np.zeros(self.state_shape, dtype=np.float32)
        
        # Snake body (channel 0)
        for x, y in self.snake[1:]:
            state[x, y, 0] = 1.0
        
        # Snake head (channel 1)
        if self.snake:
            head_x, head_y = self.snake[0]
            state[head_x, head_y, 1] = 1.0
        
        # Food (channel 2)
        if self.food:
            food_x, food_y = self.food
            state[food_x, food_y, 2] = 1.0
        
        return state
    
    def get_simple_state(self) -> np.ndarray:
        """Get simplified state for neural network (flattened)"""
        state = self._get_state()
        return state.flatten()
    
    def get_action_size(self) -> int:
        """Get action space size"""
        return self.action_size
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Get state space shape"""
        return self.state_shape
    
    def get_simple_state_size(self) -> int:
        """Get flattened state size"""
        return np.prod(self.state_shape)
    
    def get_state_size(self) -> int:
        """Get state space size (for base class compatibility)"""
        return self.get_simple_state_size()
    
    def render(self):
        """Render the game with Pygame"""
        if not self.render_mode or not PYGAME_AVAILABLE or self.screen is None:
            return
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                             self.cell_size, self.cell_size)
            color = self.colors['snake_head'] if i == 0 else self.colors['snake_body']
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)  # Border
        
        # Draw food
        if self.food:
            x, y = self.food
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors['food'], rect)
        
        # Draw score and info
        score_text = self.font.render(f"Score: {self.score}", True, self.colors['text'])
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.colors['text'])
        
        game_area_height = self.height * self.cell_size
        self.screen.blit(score_text, (10, game_area_height + 10))
        self.screen.blit(steps_text, (200, game_area_height + 10))
        
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS for visibility
    
    def close(self):
        """Close the environment"""
        if PYGAME_AVAILABLE and pygame.get_init():
            pygame.quit()
    
    def get_board_string(self) -> str:
        """Get text representation of the board"""
        board = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Place snake
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                board[y][x] = 'H'  # Head
            else:
                board[y][x] = 'S'  # Body
        
        # Place food
        if self.food:
            x, y = self.food
            board[y][x] = 'F'
        
        # Convert to string
        result = f"Score: {self.score}, Steps: {self.steps}\n"
        for row in board:
            result += ''.join(row) + '\n'
        
        return result


class DummySnakeEnv(Environment):
    """Dummy Snake environment for when Pygame is not available"""
    
    def __init__(self, width: int = 20, height: int = 20, render_mode: bool = False):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.state_shape = (width, height, 3)
        self.action_size = 4
        self.step_count = 0
        self.max_steps = 1000
        self.score = 0
        
        print("[WARNING] Using dummy Snake environment (Pygame not available)")
    
    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.score = 0
        return np.random.rand(*self.state_shape).astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        
        # Dummy dynamics
        state = np.random.rand(*self.state_shape).astype(np.float32)
        reward = np.random.uniform(-1, 2)
        done = self.step_count >= self.max_steps or np.random.random() < 0.01
        
        if reward > 1:
            self.score += 1
        
        info = {'score': self.score, 'steps': self.step_count}
        
        return state, reward, done, info
    
    def get_simple_state(self) -> np.ndarray:
        return np.random.rand(np.prod(self.state_shape)).astype(np.float32)
    
    def get_action_size(self) -> int:
        return self.action_size
    
    def get_state_shape(self) -> Tuple[int, ...]:
        return self.state_shape
    
    def get_simple_state_size(self) -> int:
        return np.prod(self.state_shape)
    
    def get_state_size(self) -> int:
        """Get state space size (for base class compatibility)"""
        return self.get_simple_state_size()
    
    def close(self):
        pass
    
    def render(self):
        if self.render_mode:
            print(f"Dummy Snake render: step {self.step_count}, score {self.score}")


def create_snake_environment(width: int = 20, height: int = 20, render_mode: bool = False) -> Environment:
    """Factory function to create Snake environment"""
    if PYGAME_AVAILABLE:
        return SnakeEnvironment(width=width, height=height, render_mode=render_mode)
    else:
        return DummySnakeEnv(width=width, height=height, render_mode=render_mode)