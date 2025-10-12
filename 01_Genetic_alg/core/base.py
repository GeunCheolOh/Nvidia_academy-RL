"""
강화학습 기본 클래스들
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple, List


class Agent(ABC):
    """강화학습 에이전트 기본 클래스"""
    
    def __init__(self, action_size: int):
        self.action_size = action_size
        self.fitness = 0.0
        
    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """상태를 받아 행동을 결정"""
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """경험을 통해 학습"""
        pass
    
    def reset(self):
        """에이전트 상태 초기화"""
        self.fitness = 0.0


class Environment(ABC):
    """환경 기본 클래스"""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """환경 초기화 및 초기 상태 반환"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """행동 실행 및 결과 반환 (next_state, reward, done, info)"""
        pass
    
    @abstractmethod
    def render(self):
        """환경 시각화"""
        pass
    
    @abstractmethod
    def get_state_size(self) -> int:
        """상태 공간 크기 반환"""
        pass
    
    @abstractmethod
    def get_action_size(self) -> int:
        """행동 공간 크기 반환"""
        pass


class Algorithm(ABC):
    """강화학습 알고리즘 기본 클래스"""
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        
    @abstractmethod
    def create_agent(self) -> Agent:
        """새로운 에이전트 생성"""
        pass
    
    @abstractmethod
    def train(self, environment: Environment, episodes: int) -> List[float]:
        """훈련 실행 및 성과 반환"""
        pass
    
    @abstractmethod
    def get_best_agent(self) -> Agent:
        """최고 성능 에이전트 반환"""
        pass


class Trainer:
    """훈련 관리 클래스"""
    
    def __init__(self, algorithm: Algorithm, environment: Environment):
        self.algorithm = algorithm
        self.environment = environment
        self.training_history = []
        
    def train(self, episodes: int, verbose: bool = True) -> List[float]:
        """훈련 실행"""
        scores = self.algorithm.train(self.environment, episodes)
        self.training_history.extend(scores)
        
        if verbose:
            print(f"훈련 완료: {episodes} 에피소드")
            print(f"평균 점수: {np.mean(scores[-100:]):.2f}")
            print(f"최고 점수: {np.max(scores):.2f}")
            
        return scores
    
    def test(self, episodes: int = 10) -> List[float]:
        """테스트 실행"""
        agent = self.algorithm.get_best_agent()
        scores = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                state, reward, done, _ = self.environment.step(action)
                total_reward += reward
                self.environment.render()
                
            scores.append(total_reward)
            print(f"테스트 {episode + 1}: 점수 {total_reward}")
            
        return scores
