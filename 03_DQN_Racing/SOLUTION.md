# DQN CarRacing 학습 과제 정답

이 문서는 DQN CarRacing 학습 과제의 정답 코드와 상세 설명을 담고 있습니다. 스스로 구현해본 후 확인용으로 사용하세요.

---

## 1. DQN 네트워크 Forward Pass

### 문제

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through network.
    
    Args:
        x: Input tensor (batch_size, channels, height, width)
        
    Returns:
        Q-values for each action
    """
    # TODO: DQN 네트워크의 forward pass를 구현하세요
    # 힌트 1: self._forward_conv(x)로 Conv 레이어를 통과시킵니다
    # 힌트 2: F.relu를 사용하여 fc1 레이어를 통과시킵니다  
    # 힌트 3: fc2 레이어를 통과시켜 최종 Q-값들을 출력합니다
    # 힌트 4: Q-값은 각 행동의 예상 가치를 나타냅니다
    #YOUR CODE HERE
    raise NotImplementedError("DQN forward pass를 구현하세요")
```

### 정답

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through network.
    
    Args:
        x: Input tensor (batch_size, channels, height, width)
        
    Returns:
        Q-values for each action
    """
    x = self._forward_conv(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
```

### 설명

**1. Convolutional 레이어 통과 (`self._forward_conv(x)`)**
```python
def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through conv layers only."""
    x = F.relu(self.conv1(x))  # 4x84x84 → 32x19x19
    x = F.relu(self.conv2(x))  # 32x19x19 → 64x8x8
    x = F.relu(self.conv3(x))  # 64x8x8 → 64x6x6
    return x.view(x.size(0), -1)  # Flatten: 64x6x6 → 2304
```
- **입력**: 4개 프레임 스택 (4, 84, 84)
- **Conv1**: 8x8 커널, stride 4 → 공간적 특징 추출
- **Conv2**: 4x4 커널, stride 2 → 고수준 특징 추출
- **Conv3**: 3x3 커널, stride 1 → 세밀한 특징 추출
- **Flatten**: 2D feature map을 1D 벡터로 변환

**2. Fully Connected 레이어 1 (`F.relu(self.fc1(x))`)**
```python
x = F.relu(self.fc1(x))  # 2304 → 512
```
- **목적**: 고차원 특징을 압축하고 비선형성 추가
- **ReLU**: 음수 값을 0으로, 양수는 그대로 (비선형 활성화)
- **512 units**: 적절한 표현력과 계산 효율의 균형

**3. Fully Connected 레이어 2 (`self.fc2(x)`)**
```python
x = self.fc2(x)  # 512 → 4 (Q-values)
```
- **출력**: 4개 행동에 대한 Q-값
  - Q[0]: 왼쪽으로 회전
  - Q[1]: 직진
  - Q[2]: 오른쪽으로 회전
  - Q[3]: 브레이크
- **활성화 함수 없음**: Q-값은 실수 범위 전체 사용

**왜 마지막에 활성화 함수가 없을까?**
- Q-값은 미래 보상의 기대값 → 음수/양수 모두 가능
- Softmax를 쓰면 확률로 변환되어 Q-값의 의미 상실
- Sigmoid를 쓰면 0~1 범위로 제한되어 표현력 감소

---

## 2. Experience Replay Buffer Sampling

### 문제

```python
def sample(self, batch_size: int) -> Tuple:
    """Sample batch of transitions."""
    # TODO: Replay Buffer에서 배치를 샘플링하세요
    # 힌트 1: random.sample을 사용하여 buffer에서 batch_size만큼 샘플링합니다
    # 힌트 2: zip(*batch)로 states, actions, rewards, next_states, dones를 분리합니다
    # 힌트 3: 각각을 적절한 torch 텐서로 변환합니다 (FloatTensor, LongTensor, BoolTensor)
    # 힌트 4: Experience Replay는 샘플 간 상관관계를 줄여 학습을 안정화합니다
    #YOUR CODE HERE
    raise NotImplementedError("Replay Buffer sampling을 구현하세요")
```

### 정답

```python
def sample(self, batch_size: int) -> Tuple:
    """Sample batch of transitions."""
    batch = random.sample(self.buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    return (
        torch.FloatTensor(np.array(states)),
        torch.LongTensor(actions),
        torch.FloatTensor(rewards),
        torch.FloatTensor(np.array(next_states)),
        torch.BoolTensor(dones)
    )
```

### 설명

**1. 무작위 샘플링 (`random.sample`)**
```python
batch = random.sample(self.buffer, batch_size)
```
- **self.buffer**: deque에 저장된 경험들 [(s,a,r,s',d), ...]
- **random.sample**: 중복 없이 batch_size개 무작위 선택
- **효과**: 연속된 샘플 간 상관관계 제거

**왜 무작위 샘플링이 중요한가?**

❌ **순차적 샘플링의 문제점**:
```python
# 연속된 프레임들만 학습
# Time 1: [frame1, frame2, frame3, frame4] → 행동: 직진
# Time 2: [frame2, frame3, frame4, frame5] → 행동: 직진
# Time 3: [frame3, frame4, frame5, frame6] → 행동: 직진
# → 매우 높은 상관관계, 학습 불안정
```

✅ **무작위 샘플링의 효과**:
```python
# 다양한 시점의 경험 혼합
# Sample 1: Episode 3의 Time 100
# Sample 2: Episode 7의 Time 50
# Sample 3: Episode 1의 Time 200
# → 낮은 상관관계, 안정적 학습
```

**2. 데이터 언패킹 (`zip(*batch)`)**
```python
states, actions, rewards, next_states, dones = zip(*batch)
```

**동작 원리**:
```python
# batch = [
#     (s1, a1, r1, s1', d1),
#     (s2, a2, r2, s2', d2),
#     (s3, a3, r3, s3', d3)
# ]

# zip(*batch)는 다음과 같이 변환:
# states = (s1, s2, s3)
# actions = (a1, a2, a3)
# rewards = (r1, r2, r3)
# next_states = (s1', s2', s3')
# dones = (d1, d2, d3)
```

**3. 텐서 변환**

```python
# States: (batch_size, 4, 84, 84) 이미지 데이터
torch.FloatTensor(np.array(states))

# Actions: (batch_size,) 행동 인덱스 (0~3)
torch.LongTensor(actions)

# Rewards: (batch_size,) 보상 값 (실수)
torch.FloatTensor(rewards)

# Next States: (batch_size, 4, 84, 84) 다음 이미지
torch.FloatTensor(np.array(next_states))

# Dones: (batch_size,) 종료 여부 (True/False)
torch.BoolTensor(dones)
```

**텐서 타입 선택 이유**:
- **FloatTensor**: 연속 값 (이미지, 보상) → 그래디언트 계산 가능
- **LongTensor**: 정수 인덱스 (행동) → gather 연산용
- **BoolTensor**: 논리 값 (종료 여부) → 마스킹용

---

## 3. Epsilon-Greedy Action Selection

### 문제

```python
def select_action(self, state: np.ndarray, training: bool = True) -> int:
    """
    Select action using epsilon-greedy policy.
    
    Args:
        state: Current state
        training: Whether in training mode
        
    Returns:
        Selected action
    """
    # TODO: Epsilon-greedy 정책으로 행동을 선택하세요
    # 힌트 1: training이 True이고 random.random() < epsilon이면 무작위 행동 선택 (탐험)
    # 힌트 2: 그렇지 않으면 main_network로 Q-값을 계산하여 최대값의 행동 선택 (활용)
    # 힌트 3: 추론 시에는 torch.no_grad()를 사용하여 그래디언트 계산을 방지합니다
    # 힌트 4: 텐서를 device로 이동시키고, argmax()로 최대 Q-값의 인덱스를 가져옵니다
    #YOUR CODE HERE
    raise NotImplementedError("Epsilon-greedy action selection을 구현하세요")
```

### 정답

```python
def select_action(self, state: np.ndarray, training: bool = True) -> int:
    """
    Select action using epsilon-greedy policy.
    
    Args:
        state: Current state
        training: Whether in training mode
        
    Returns:
        Selected action
    """
    if training and random.random() < self.epsilon:
        return random.randint(0, self.action_dim - 1)
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.main_network(state_tensor)
        return q_values.argmax().item()
```

### 설명

**1. 탐험 (Exploration)**
```python
if training and random.random() < self.epsilon:
    return random.randint(0, self.action_dim - 1)
```

- **조건**: 학습 모드이고 랜덤 값이 epsilon보다 작을 때
- **행동**: 0~3 사이 무작위 행동 선택
- **목적**: 새로운 상태-행동 쌍 탐색, 지역 최적해 탈출

**Epsilon 감쇠 전략**:
```python
# 학습 초기 (Episode 0)
epsilon = 1.0  # 100% 탐험

# 중기 (Episode 100)
epsilon = 1.0 * (0.995^100) ≈ 0.606  # 60% 탐험

# 후기 (Episode 500)
epsilon = 1.0 * (0.995^500) ≈ 0.082  # 8% 탐험

# 최종
epsilon = max(0.01, current)  # 최소 1% 탐험 유지
```

**2. 활용 (Exploitation)**
```python
with torch.no_grad():
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    q_values = self.main_network(state_tensor)
    return q_values.argmax().item()
```

**단계별 분석**:

**a) `torch.no_grad()`**
```python
with torch.no_grad():
    # 이 블록 안에서는 그래디언트 계산 안 함
```
- **메모리 절약**: 그래디언트 저장 공간 불필요
- **속도 향상**: 연산 그래프 생성 생략
- **추론 전용**: 학습이 아닌 행동 선택만 수행

**b) 텐서 변환 및 배치 차원 추가**
```python
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

# state: (4, 84, 84) → numpy array
# FloatTensor(state): (4, 84, 84) → torch tensor
# unsqueeze(0): (1, 4, 84, 84) → 배치 차원 추가
# to(device): GPU로 이동 (가능한 경우)
```

**왜 배치 차원이 필요한가?**
- 네트워크는 배치 입력을 기대: `(batch_size, channels, H, W)`
- 단일 샘플도 배치 형태로: `(1, 4, 84, 84)`

**c) Q-값 계산 및 행동 선택**
```python
q_values = self.main_network(state_tensor)
# 출력: tensor([[Q_left, Q_straight, Q_right, Q_brake]])
#       shape: (1, 4)

return q_values.argmax().item()
# argmax(): 최대값의 인덱스 → tensor(2) if right has max Q
# item(): Python int로 변환 → 2
```

**예시**:
```python
# Q-값: [-0.5, 0.3, 0.8, -0.2]
#        left  straight right brake
# argmax() → 2 (right가 최대)
# 행동: 오른쪽 회전
```

---

## 4. DQN Update (Bellman Equation)

### 문제

```python
def update(self) -> Optional[float]:
    """
    Update network using batch from replay buffer.
    
    Returns:
        Loss value if update performed, None otherwise
    """
    if len(self.replay_buffer) < HYPERPARAMETERS['batch_size']:
        return None
        
    # Sample batch
    states, actions, rewards, next_states, dones = \
        self.replay_buffer.sample(HYPERPARAMETERS['batch_size'])
        
    states = states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    next_states = next_states.to(self.device)
    dones = dones.to(self.device)
    
    # TODO: DQN 학습 업데이트를 구현하세요 (Bellman Equation)
    # 힌트 1: main_network로 현재 상태의 Q-값을 계산하고 gather로 선택한 행동의 Q-값 추출
    # 힌트 2: target_network로 다음 상태의 최대 Q-값을 계산 (torch.no_grad() 사용)
    # 힌트 3: Target = reward + gamma * max Q(next_state) * (에피소드가 끝나지 않았으면)
    # 힌트 4: Loss = smooth_l1_loss(current_Q, target)로 손실 계산
    # 힌트 5: optimizer.zero_grad() → loss.backward() → clip_grad_norm → optimizer.step()
    # 힌트 6: 일정 스텝마다 target network를 main network로 업데이트
    #YOUR CODE HERE
    raise NotImplementedError("DQN update를 구현하세요")
```

### 정답

```python
def update(self) -> Optional[float]:
    """
    Update network using batch from replay buffer.
    
    Returns:
        Loss value if update performed, None otherwise
    """
    if len(self.replay_buffer) < HYPERPARAMETERS['batch_size']:
        return None
        
    # Sample batch
    states, actions, rewards, next_states, dones = \
        self.replay_buffer.sample(HYPERPARAMETERS['batch_size'])
        
    states = states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    next_states = next_states.to(self.device)
    dones = dones.to(self.device)
    
    # Current Q-values
    current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1))
    
    # Next Q-values from target network
    with torch.no_grad():
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + (HYPERPARAMETERS['gamma'] * next_q_values * (~dones))
        
    # Compute loss
    loss = F.smooth_l1_loss(current_q_values.squeeze(), targets)
    
    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
    self.optimizer.step()
    
    # Update step counter
    self.step_count += 1
    
    # Update target network
    if self.step_count % HYPERPARAMETERS['target_update'] == 0:
        self.update_target_network()
        
    return loss.item()
```

### 설명

**1. 현재 Q-값 계산**
```python
current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1))
```

**단계별 분석**:
```python
# states: (32, 4, 84, 84) - 배치 크기 32
q_all = self.main_network(states)
# q_all: (32, 4) - 각 샘플의 4개 행동에 대한 Q-값

# actions: (32,) - [2, 1, 3, 0, ...] 각 샘플의 선택된 행동
# actions.unsqueeze(1): (32, 1) - gather를 위한 차원 추가

current_q_values = q_all.gather(1, actions.unsqueeze(1))
# (32, 1) - 각 샘플에서 선택한 행동의 Q-값만 추출
```

**gather 동작 예시**:
```python
# q_all = [[0.5, 0.3, 0.8, 0.1],   # 샘플 0
#          [0.2, 0.6, 0.4, 0.3]]   # 샘플 1
# actions = [2, 1]  # 샘플 0은 행동 2, 샘플 1은 행동 1 선택

# gather 결과:
# [[0.8],  # 샘플 0의 행동 2의 Q-값
#  [0.6]]  # 샘플 1의 행동 1의 Q-값
```

**2. Target Q-값 계산 (Bellman Equation)**
```python
with torch.no_grad():
    next_q_values = self.target_network(next_states).max(1)[0]
    targets = rewards + (HYPERPARAMETERS['gamma'] * next_q_values * (~dones))
```

**Target Network 사용 이유**:
```
Main Network (학습 중):
    Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                               ↑
                         여기서 Q를 계산

문제: Q가 계속 변하면 target도 계속 변함 → 불안정

해결: Target Network (고정된 Q)를 사용
    Q_main ← Q_main + α[r + γ max Q_target(s',a') - Q_main]
    
    Target Network는 1000 스텝마다만 업데이트
```

**다음 Q-값 계산**:
```python
next_q_values = self.target_network(next_states).max(1)[0]

# next_states: (32, 4, 84, 84)
# target_network(next_states): (32, 4)
# max(1): 차원 1 (행동 차원)에서 최대값
#   [0]: 최대값만 가져오기 (인덱스는 버림)
# 결과: (32,) - 각 샘플의 최대 Q-값
```

**Target 계산 (종료 처리 포함)**:
```python
targets = rewards + (gamma * next_q_values * (~dones))

# rewards: (32,)
# gamma: 0.99
# next_q_values: (32,)
# ~dones: (32,) - Boolean not (True→False, False→True)

# 에피소드가 끝나지 않았을 때 (done=False):
# target = r + 0.99 * max Q(s',a') * True(1)
# target = r + 0.99 * max Q(s',a')

# 에피소드가 끝났을 때 (done=True):
# target = r + 0.99 * max Q(s',a') * False(0)
# target = r
```

**왜 종료 시 next_q_values를 0으로?**
- 에피소드 종료 = 더 이상 미래 보상 없음
- Target = 즉시 받은 보상만 사용
- 예: 구멍에 빠짐 → 보상 0, 미래 없음 → Target = 0

**3. 손실 함수 (Smooth L1 Loss)**
```python
loss = F.smooth_l1_loss(current_q_values.squeeze(), targets)
```

**Smooth L1 Loss (Huber Loss)**:
```python
L(x) = {
    0.5 * x^2          if |x| < 1
    |x| - 0.5          if |x| >= 1
}

# 장점:
# 1. 작은 오차: L2처럼 부드러운 그래디언트 (x^2)
# 2. 큰 오차: L1처럼 outlier에 robust (|x|)
```

**왜 MSE 대신 Smooth L1?**
```
MSE Loss: (predict - target)^2
- 장점: 미분 가능, 볼록 함수
- 단점: 큰 오차에 민감 (제곱으로 증폭)

Smooth L1 Loss:
- 작은 오차: MSE처럼 정확
- 큰 오차: MSE보다 완만 → 학습 안정
```

**4. 최적화 단계**

**a) Gradient 초기화**
```python
self.optimizer.zero_grad()
```
- 이전 step의 gradient를 0으로 초기화
- PyTorch는 gradient를 누적하므로 매번 초기화 필요

**b) Backpropagation**
```python
loss.backward()
```
- 손실에서 모든 파라미터로 그래디언트 계산
- Chain rule로 역전파

**c) Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
```
- **목적**: Gradient exploding 방지
- **방법**: 그래디언트 norm이 1.0을 넘으면 스케일 다운

**Gradient Clipping 동작**:
```python
# Before clipping:
# grad_conv1 = [10.5, -8.2, 15.3, ...]
# grad_fc1 = [7.8, -12.1, 9.5, ...]
# total_norm = sqrt(10.5^2 + 8.2^2 + ... + 9.5^2) = 3.5

# max_norm = 1.0이므로 스케일 다운
# scale = 1.0 / 3.5 = 0.286

# After clipping:
# grad_conv1 = [3.0, -2.3, 4.4, ...]  (모두 0.286배)
# grad_fc1 = [2.2, -3.5, 2.7, ...]
# total_norm = 1.0
```

**d) 파라미터 업데이트**
```python
self.optimizer.step()
```
- Adam optimizer로 파라미터 업데이트
- θ ← θ - lr * gradient (단순화)

**5. Target Network 업데이트**
```python
self.step_count += 1

if self.step_count % HYPERPARAMETERS['target_update'] == 0:
    self.update_target_network()
```

```python
def update_target_network(self):
    """Update target network with main network weights."""
    self.target_network.load_state_dict(self.main_network.state_dict())
```

- **주기**: 1000 스텝마다 (HYPERPARAMETERS['target_update'])
- **방법**: Main network의 가중치를 Target network에 복사
- **효과**: Target이 천천히 변하여 학습 안정화

**Hard Update vs Soft Update**:
```python
# Hard Update (현재 구현)
θ_target ← θ_main  (1000 스텝마다)

# Soft Update (대안)
θ_target ← τ*θ_main + (1-τ)*θ_target  (매 스텝)
# τ = 0.001 정도
```

---

## 핵심 개념 요약

### DQN의 3대 핵심 기술

**1. Experience Replay**
```python
# 문제: 연속된 샘플의 높은 상관관계
Episode t:   [s1, a1, r1, s2] → [s2, a2, r2, s3] → [s3, a3, r3, s4]
             매우 비슷한 경험들 → 학습 불안정

# 해결: 버퍼에 저장 후 무작위 샘플링
Buffer: [(s1,a1,r1,s2), (s2,a2,r2,s3), ..., (s100,a100,r100,s101)]
Sample: [(s37,a37,r37,s38), (s82,a82,r82,s83), (s15,a15,r15,s16)]
        독립적인 경험들 → 학습 안정
```

**2. Target Network**
```python
# 문제: Moving target
Q ← Q + α[r + γ max Q(s') - Q]
          ↑ Q가 변하면 target도 변함

# 해결: 고정된 target 사용
Q_main ← Q_main + α[r + γ max Q_target(s') - Q_main]
Q_target는 1000 스텝마다만 업데이트
```

**3. CNN Feature Extraction**
```
Raw pixels (96x96x3) → 너무 고차원
           ↓ CNN
      Features (512) → 압축된 표현
           ↓ FC
     Q-values (4) → 행동 가치
```

### Bellman Equation의 의미

```
Q(s,a) = E[r + γ max Q(s',a')]

"현재 상태 s에서 행동 a를 했을 때의 가치는
 즉시 받는 보상 r과
 다음 상태 s'에서 최선의 행동을 했을 때의 할인된 가치의 합"
```

**예시**:
```python
# CarRacing에서
# 현재: 트랙을 잘 따라가는 중
# 행동: 오른쪽 회전
# 보상: +0.5 (트랙 유지)

# Q(현재, 오른쪽) = 0.5 + 0.99 * max Q(다음상태, 모든행동)
#                 = 0.5 + 0.99 * max[Q(다음, 왼쪽),
#                                    Q(다음, 직진),
#                                    Q(다음, 오른쪽),
#                                    Q(다음, 브레이크)]
#                 = 0.5 + 0.99 * 2.3
#                 = 2.777
```

---

## 디버깅 가이드

### 자주 발생하는 오류

**1. RuntimeError: expected scalar type Float but found Double**
```python
# 문제
state = np.array(...)  # dtype=float64
q_values = network(torch.tensor(state))  # 오류!

# 해결
q_values = network(torch.FloatTensor(state))  # float32
```

**2. RuntimeError: Sizes of tensors must match**
```python
# 문제: gather 차원 불일치
actions = torch.LongTensor([0, 1, 2])  # (3,)
q_values.gather(1, actions)  # 오류! 차원 불일치

# 해결
q_values.gather(1, actions.unsqueeze(1))  # (3, 1)
```

**3. AssertionError: buffer size < batch_size**
```python
# 문제: 버퍼가 충분히 차지 않음
if len(buffer) < 32:  # 학습 안 함
    return None

# 정상: 처음 32개 경험을 모을 때까지 기다림
```

**4. 학습이 전혀 안 되는 경우**
```python
# 원인 1: Epsilon이 1.0으로 고정
# → 무작위 행동만, 학습 안 됨
# 해결: epsilon decay 확인

# 원인 2: Target network 업데이트 안 됨
# → target이 고정되어 학습 안 됨
# 해결: step_count 증가 및 업데이트 로직 확인

# 원인 3: Gradient clipping이 너무 작음
# → gradient가 0으로 수렴
# 해결: max_norm을 1.0 정도로 설정
```

**5. Loss가 발산하는 경우**
```python
# 원인 1: Learning rate가 너무 큼
# 해결: 0.0001 → 0.00001

# 원인 2: Gradient clipping 없음
# 해결: clip_grad_norm_(model.parameters(), 1.0)

# 원인 3: Target network 없음
# 해결: Target network 구현 확인
```

---

## 성능 개선 팁

### 1. 하이퍼파라미터 튜닝

**Learning Rate**
```python
# 너무 크면: 발산
lr = 0.001  # ❌ 불안정

# 너무 작으면: 느린 학습
lr = 0.00001  # ❌ 너무 느림

# 적절한 값
lr = 0.0001  # ✅ 좋은 시작점
```

**Epsilon Decay**
```python
# 너무 빠르면: 조기 수렴
epsilon_decay = 0.99  # ❌ 너무 빠름

# 너무 느리면: 오래 탐험
epsilon_decay = 0.999  # ❌ 너무 느림

# 적절한 값
epsilon_decay = 0.995  # ✅ 균형적
```

**Replay Buffer Size**
```python
# 너무 작으면: 샘플 다양성 부족
buffer_size = 1000  # ❌ 작음

# 너무 크면: 메모리 부족
buffer_size = 1000000  # ❌ 클 수 있음

# 적절한 값
buffer_size = 10000  # ✅ 좋은 시작점
```

### 2. 네트워크 구조 개선

**더 깊은 네트워크**
```python
# 기본
Conv1 → Conv2 → Conv3 → FC1 → FC2

# 개선
Conv1 → Conv2 → Conv3 → Conv4 → FC1 → FC2 → FC3
```

**Dueling DQN**
```python
# 기본 DQN
Features → Q(s,a)

# Dueling DQN
Features → Value(s) ─┐
                      ├→ Q(s,a) = V(s) + A(s,a)
Features → Advantage(s,a) ─┘
```

### 3. 학습 전략

**Frame Skipping**
```python
# 같은 행동을 4번 반복
for _ in range(4):
    next_state, reward, done, _ = env.step(action)
    
# 효과: 학습 속도 4배, 반응성은 약간 감소
```

**Reward Shaping**
```python
# 기본: 트랙 유지 시에만 보상
reward = 0 or -0.1

# 개선: 진행도 보상 추가
reward = speed * on_track_bonus + off_track_penalty
```

---

## 확장 과제

### 1. Double DQN
```python
# 기본 DQN
target = r + γ max_a Q_target(s', a)

# Double DQN
a* = argmax_a Q_main(s', a)  # Main으로 행동 선택
target = r + γ Q_target(s', a*)  # Target으로 가치 평가
```

### 2. Prioritized Experience Replay
```python
# 기본: 균등 샘플링
batch = random.sample(buffer, batch_size)

# PER: TD error에 비례한 샘플링
priority = |TD_error| + ε
prob = priority^α / Σ priority^α
batch = sample_with_prob(buffer, prob, batch_size)
```

### 3. Multi-step Returns
```python
# 1-step
target = r_t + γ max Q(s_{t+1})

# n-step
target = r_t + γr_{t+1} + γ^2r_{t+2} + ... + γ^n max Q(s_{t+n})
```

---

이 정답 파일은 학습 과정에서 막혔을 때 참고용으로 사용하세요.
가능한 한 스스로 구현해보고, 어려운 부분만 힌트로 활용하는 것을 권장합니다.

