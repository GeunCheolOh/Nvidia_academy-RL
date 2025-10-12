"""
개선된 유전 알고리즘 구현
"""
import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple
from core.base import Agent, Algorithm, Environment


class NeuralNetwork:
    """신경망 클래스"""
    
    def __init__(self, layers: List[int], activation='relu'):
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # 가중치와 편향 초기화 (Xavier 초기화)
        for i in range(len(layers) - 1):
            limit = np.sqrt(6.0 / (layers[i] + layers[i + 1]))
            weight = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            bias = np.zeros((1, layers[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """순전파"""
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, weight) + bias
            
            # 마지막 층이 아니면 활성화 함수 적용
            if i < len(self.weights) - 1:
                if self.activation == 'relu':
                    x = np.maximum(0, x)
                elif self.activation == 'tanh':
                    x = np.tanh(x)
                elif self.activation == 'sigmoid':
                    x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        # 출력층에는 소프트맥스 적용
        return self._softmax(x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """소프트맥스 함수"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def get_weights_as_vector(self) -> np.ndarray:
        """모든 가중치를 1차원 벡터로 변환"""
        vector = np.array([])
        for weight in self.weights:
            vector = np.concatenate([vector, weight.flatten()])
        for bias in self.biases:
            vector = np.concatenate([vector, bias.flatten()])
        return vector
    
    def set_weights_from_vector(self, vector: np.ndarray):
        """1차원 벡터에서 가중치 복원"""
        start_idx = 0
        
        for i, weight in enumerate(self.weights):
            size = weight.size
            self.weights[i] = vector[start_idx:start_idx + size].reshape(weight.shape)
            start_idx += size
        
        for i, bias in enumerate(self.biases):
            size = bias.size
            self.biases[i] = vector[start_idx:start_idx + size].reshape(bias.shape)
            start_idx += size


class GeneticAgent(Agent):
    """유전 알고리즘 에이전트"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = None):
        super().__init__(action_size)
        
        if hidden_layers is None:
            hidden_layers = [32, 16]
        
        layers = [state_size] + hidden_layers + [action_size]
        self.network = NeuralNetwork(layers)
        self.fitness_history = []
    
    def act(self, state: np.ndarray) -> int:
        """상태를 받아 행동 선택"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        action_probs = self.network.forward(state)
        return np.argmax(action_probs)
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """유전 알고리즘에서는 개별 경험으로 학습하지 않음"""
        pass
    
    def set_weights(self, weights: np.ndarray):
        """가중치 설정"""
        self.network.set_weights_from_vector(weights)
    
    def get_weights(self) -> np.ndarray:
        """가중치 반환"""
        return self.network.get_weights_as_vector()
    
    def mutate(self, mutation_rate: float, mutation_strength: float):
        """돌연변이 적용"""
        weights = self.get_weights()
        
        # 가우시안 노이즈를 이용한 돌연변이
        mask = np.random.random(weights.shape) < mutation_rate
        noise = np.random.normal(0, mutation_strength, weights.shape)
        weights[mask] += noise[mask]
        
        self.set_weights(weights)


class GeneticAlgorithm(Algorithm):
    """개선된 유전 알고리즘"""
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 population_size: int = 100,
                 elite_size: int = 20,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.1,
                 crossover_rate: float = 0.8,
                 selection_method: str = 'tournament',
                 hidden_layers: List[int] = None):
        
        super().__init__(state_size, action_size)
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.hidden_layers = hidden_layers or [32, 16]
        
        # 개체군 초기화
        self.population = []
        for _ in range(population_size):
            agent = GeneticAgent(state_size, action_size, hidden_layers)
            self.population.append(agent)
        
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def create_agent(self) -> GeneticAgent:
        """새로운 에이전트 생성"""
        return GeneticAgent(self.state_size, self.action_size, self.hidden_layers)
    
    def evaluate_population(self, environment: Environment) -> List[float]:
        """개체군 평가"""
        fitness_scores = []
        
        for i, agent in enumerate(self.population):
            total_fitness = 0
            num_episodes = 3  # 각 에이전트를 여러 번 평가
            
            for episode in range(num_episodes):
                state = environment.reset()
                episode_reward = 0
                steps = 0
                max_steps = 1000
                
                while steps < max_steps:
                    action = agent.act(state)
                    next_state, reward, done, info = environment.step(action)
                    
                    episode_reward += reward
                    steps += 1
                    
                    if done:
                        break
                    
                    state = next_state
                
                # 생존 시간도 보상에 포함
                survival_bonus = min(steps / max_steps * 50, 50)
                total_fitness += episode_reward + survival_bonus
            
            agent.fitness = total_fitness / num_episodes
            fitness_scores.append(agent.fitness)
        
        return fitness_scores
    
    def selection(self, k: int = 3) -> List[GeneticAgent]:
        """선택 연산"""
        if self.selection_method == 'tournament':
            return self._tournament_selection(k)
        elif self.selection_method == 'roulette':
            return self._roulette_selection()
        elif self.selection_method == 'rank':
            return self._rank_selection()
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _tournament_selection(self, k: int) -> List[GeneticAgent]:
        """토너먼트 선택"""
        selected = []
        
        for _ in range(self.population_size - self.elite_size):
            tournament = random.sample(self.population, k)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(deepcopy(winner))
        
        return selected
    
    def _roulette_selection(self) -> List[GeneticAgent]:
        """룰렛 휠 선택"""
        # 음수 피트니스 처리
        min_fitness = min(agent.fitness for agent in self.population)
        adjusted_fitness = [agent.fitness - min_fitness + 1 for agent in self.population]
        total_fitness = sum(adjusted_fitness)
        
        selected = []
        for _ in range(self.population_size - self.elite_size):
            pick = random.uniform(0, total_fitness)
            current = 0
            
            for i, agent in enumerate(self.population):
                current += adjusted_fitness[i]
                if current >= pick:
                    selected.append(deepcopy(agent))
                    break
        
        return selected
    
    def _rank_selection(self) -> List[GeneticAgent]:
        """순위 선택"""
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        ranks = list(range(len(sorted_population), 0, -1))
        total_rank = sum(ranks)
        
        selected = []
        for _ in range(self.population_size - self.elite_size):
            pick = random.uniform(0, total_rank)
            current = 0
            
            for i, agent in enumerate(sorted_population):
                current += ranks[i]
                if current >= pick:
                    selected.append(deepcopy(agent))
                    break
        
        return selected
    
    def crossover(self, parent1: GeneticAgent, parent2: GeneticAgent) -> Tuple[GeneticAgent, GeneticAgent]:
        """교차 연산 (균등 교차)"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # 균등 교차
        mask = np.random.random(weights1.shape) < 0.5
        
        child1_weights = np.where(mask, weights1, weights2)
        child2_weights = np.where(mask, weights2, weights1)
        
        child1 = self.create_agent()
        child2 = self.create_agent()
        
        child1.set_weights(child1_weights)
        child2.set_weights(child2_weights)
        
        return child1, child2
    
    def evolve_population(self):
        """개체군 진화"""
        # 엘리트 선택
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population = deepcopy(sorted_population[:self.elite_size])
        
        # 선택
        selected = self.selection()
        
        # 교차 및 돌연변이
        while len(new_population) < self.population_size:
            if len(selected) >= 2:
                parent1 = selected.pop()
                parent2 = selected.pop()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                # 돌연변이
                child1.mutate(self.mutation_rate, self.mutation_strength)
                child2.mutate(self.mutation_rate, self.mutation_strength)
                
                new_population.extend([child1, child2])
            else:
                # 부모가 부족하면 무작위 개체 생성
                new_agent = self.create_agent()
                new_population.append(new_agent)
        
        # 개체수 조정
        self.population = new_population[:self.population_size]
    
    def train(self, environment: Environment, generations: int) -> List[float]:
        """훈련 실행"""
        best_fitness_per_generation = []
        
        for generation in range(generations):
            self.generation = generation
            
            # 개체군 평가
            fitness_scores = self.evaluate_population(environment)
            
            # 통계 계산
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            best_fitness_per_generation.append(best_fitness)
            
            print(f"세대 {generation + 1}/{generations} - "
                  f"최고 적합도: {best_fitness:.2f}, "
                  f"평균 적합도: {avg_fitness:.2f}")
            
            # 개체군 진화
            if generation < generations - 1:
                self.evolve_population()
                
                # 적응적 돌연변이 (성능이 정체되면 돌연변이율 증가)
                if len(self.best_fitness_history) >= 10:
                    recent_improvement = (self.best_fitness_history[-1] - 
                                        self.best_fitness_history[-10])
                    if recent_improvement < 5:  # 개선이 없으면
                        self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
                        self.mutation_strength = min(0.3, self.mutation_strength * 1.1)
                    else:  # 개선이 있으면
                        self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
                        self.mutation_strength = max(0.05, self.mutation_strength * 0.95)
        
        return best_fitness_per_generation
    
    def get_best_agent(self) -> GeneticAgent:
        """최고 성능 에이전트 반환"""
        return max(self.population, key=lambda x: x.fitness)
