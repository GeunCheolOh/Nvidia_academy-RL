"""
Genome - 신경망 유전자
6 -> 10 -> 20 -> 10 -> 3 구조의 심층 신경망
"""
import numpy as np


class Genome:
    """유전 알고리즘을 위한 신경망 유전자"""
    
    def __init__(self):
        """신경망 가중치 초기화"""
        self.fitness = 0
        
        # 신경망 구조: 6 -> 10 -> 20 -> 10 -> 3
        hidden_layer = 10
        self.w1 = np.random.randn(6, hidden_layer)
        self.w2 = np.random.randn(hidden_layer, 20)
        self.w3 = np.random.randn(20, hidden_layer)
        self.w4 = np.random.randn(hidden_layer, 3)
    
    def forward(self, inputs):
        """
        순전파 (Forward Propagation)
        
        Args:
            inputs: 6차원 입력 벡터
            
        Returns:
            3차원 출력 (직진, 좌회전, 우회전)
        """
        # TODO 1: 첫 번째 은닉층 (6 -> 10)
        # 힌트: np.matmul(inputs, self.w1)로 행렬 곱 수행 후 relu 적용
        # net = np.matmul(inputs, self.w?)
        # net = self.relu(net)
        # YOUR CODE HERE
        raise NotImplementedError("TODO 1: 첫 번째 은닉층을 구현하세요")
        
        # TODO 2: 두 번째 은닉층 (10 -> 20)
        # 힌트: w2를 사용하여 행렬 곱 수행 후 relu 적용
        # YOUR CODE HERE
        raise NotImplementedError("TODO 2: 두 번째 은닉층을 구현하세요")
        
        # TODO 3: 세 번째 은닉층 (20 -> 10)
        # 힌트: w3를 사용하여 행렬 곱 수행 후 relu 적용
        # YOUR CODE HERE
        raise NotImplementedError("TODO 3: 세 번째 은닉층을 구현하세요")
        
        # TODO 4: 출력층 (10 -> 3)
        # 힌트: w4를 사용하여 행렬 곱 수행 후 softmax 적용
        # YOUR CODE HERE
        raise NotImplementedError("TODO 4: 출력층을 구현하세요")
        
        return net
    
    def relu(self, x):
        """
        ReLU 활성화 함수 (Rectified Linear Unit)
        
        Args:
            x: 입력 값
            
        Returns:
            max(0, x) - 음수는 0으로, 양수는 그대로
        """
        # TODO 5: ReLU 함수 구현
        # 힌트: x가 0보다 크면 x, 작으면 0을 반환
        # 방법 1: return x * (x >= 0)
        # 방법 2: return np.maximum(0, x)
        # YOUR CODE HERE
        raise NotImplementedError("TODO 5: ReLU 활성화 함수를 구현하세요")
    
    def softmax(self, x):
        """
        Softmax 활성화 함수
        출력을 확률 분포로 변환 (합이 1)
        
        Args:
            x: 입력 벡터
            
        Returns:
            확률 벡터 (각 원소는 0~1, 합은 1)
        """
        # TODO 6: Softmax 함수 구현
        # 힌트: exp(x) / sum(exp(x))
        # 수치 안정성을 위해 x에서 최댓값을 빼주세요
        # exp_x = np.exp(x - np.max(x))
        # return exp_x / np.sum(exp_x)
        # YOUR CODE HERE
        raise NotImplementedError("TODO 6: Softmax 활성화 함수를 구현하세요")
    
    def save(self, filepath):
        """유전자 저장"""
        np.savez(filepath,
                 w1=self.w1, w2=self.w2, w3=self.w3, w4=self.w4,
                 fitness=self.fitness)
    
    def load(self, filepath):
        """유전자 로드"""
        data = np.load(filepath)
        self.w1 = data['w1']
        self.w2 = data['w2']
        self.w3 = data['w3']
        self.w4 = data['w4']
        if 'fitness' in data:
            self.fitness = float(data['fitness'])

