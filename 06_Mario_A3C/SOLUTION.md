# A2C Super Mario Bros 학습 과제 정답

이 문서는 A2C Super Mario Bros 학습 과제의 정답 코드와 상세 설명을 담고 있습니다. 스스로 구현해본 후 확인용으로 사용하세요.

---

## 1. Actor-Critic 네트워크 Forward Pass

### 문제 (`training/src/model.py`)

```python
def forward(self, x, hx, cx):
    """
    Forward pass through the network.

    Args:
        x: Input state (batch_size, num_inputs, 84, 84)
        hx: Hidden state of LSTM
        cx: Cell state of LSTM

    Returns:
        Tuple of (actor_logits, critic_value, new_hx, new_cx)
    """
    # TODO: Actor-Critic 네트워크의 forward pass를 구현하세요
    # 힌트 1: Conv 레이어들(conv1~4)을 순차적으로 통과시키고 F.relu 적용
    # 힌트 2: x.view로 flatten하여 LSTM 입력 준비
    # 힌트 3: LSTM을 통과시켜 temporal dependency 학습 (hidden/cell state 업데이트)
    # 힌트 4: actor_linear로 행동 확률(logits) 출력
    # 힌트 5: critic_linear로 상태 가치(value) 출력
    #YOUR CODE HERE
    raise NotImplementedError("Actor-Critic forward pass를 구현하세요")
```

### 정답

```python
def forward(self, x, hx, cx):
    """
    Forward pass through the network.

    Args:
        x: Input state (batch_size, num_inputs, 84, 84)
        hx: Hidden state of LSTM
        cx: Cell state of LSTM

    Returns:
        Tuple of (actor_logits, critic_value, new_hx, new_cx)
    """
    # Convolutional feature extraction
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))

    # Flatten and pass through LSTM
    x = x.view(x.size(0), -1)
    hx, cx = self.lstm(x, (hx, cx))

    # Actor and Critic outputs
    actor_logits = self.actor_linear(hx)
    critic_value = self.critic_linear(hx)

    return actor_logits, critic_value, hx, cx
```

### 설명

**1. Convolutional Feature Extraction**

```python
x = F.relu(self.conv1(x))  # (4, 84, 84) → (32, 41, 41)
x = F.relu(self.conv2(x))  # (32, 41, 41) → (32, 20, 20)
x = F.relu(self.conv3(x))  # (32, 20, 20) → (32, 9, 9)
x = F.relu(self.conv4(x))  # (32, 9, 9) → (32, 6, 6)
```

**레이어별 분석**:

```python
# Conv1: 입력 채널 4 (프레임 스택) → 출력 32 채널
self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
# 3x3 커널, stride 2 → 공간 해상도 절반
# 84x84 → 41x41 (floor(84/2) = 42, padding 조정)

# Conv2~4: 동일한 구조
# 목적: 점진적으로 공간 차원 축소, 채널(특징) 유지
```

**왜 4개의 Conv 레이어?**
- **얕은 레이어 (Conv1-2)**: 엣지, 코너 등 저수준 특징
- **깊은 레이어 (Conv3-4)**: 적, 블록, 파이프 등 고수준 특징
- **Stride 2**: 연산량 감소, receptive field 증가

**2. Flatten for LSTM**

```python
x = x.view(x.size(0), -1)
# Before: (batch_size, 32, 6, 6)
# After:  (batch_size, 1152)  # 32 * 6 * 6 = 1152
```

**왜 Flatten?**
- LSTM은 1D 입력만 받음
- 2D feature map을 1D 벡터로 변환
- 공간 정보는 손실되지만 특징은 보존

**3. LSTM Layer (Temporal Dependency)**

```python
hx, cx = self.lstm(x, (hx, cx))
# x: (batch_size, 1152) - 현재 프레임 특징
# hx: (batch_size, 512) - 이전 hidden state
# cx: (batch_size, 512) - 이전 cell state
# 
# 출력:
# hx: (batch_size, 512) - 새로운 hidden state
# cx: (batch_size, 512) - 새로운 cell state
```

**LSTM의 역할**:
```python
# 시간 t=0 (게임 시작)
hx_0, cx_0 = 0, 0  # 초기 상태

# 시간 t=1 (첫 프레임)
hx_1, cx_1 = LSTM(frame_1, hx_0, cx_0)
# "마리오가 서 있다" 정보 저장

# 시간 t=2 (둘째 프레임)
hx_2, cx_2 = LSTM(frame_2, hx_1, cx_1)
# "마리오가 움직이기 시작했다" 정보 누적

# 시간 t=3 (셋째 프레임)
hx_3, cx_3 = LSTM(frame_3, hx_2, cx_2)
# "마리오가 점프 중이다" 정보 누적
```

**LSTM이 없다면?**
- 단일 프레임만 보고 판단
- "마리오가 공중에 있다" → 왜? 점프? 추락?
- 과거 정보 없이 현재만 판단 → 잘못된 행동

**LSTM이 있다면**:
- 여러 프레임의 패턴 학습
- "마리오가 점프 중 + 위로 올라가는 중" → 계속 전진
- "마리오가 점프 중 + 내려오는 중" → 착지 준비
- 시간적 맥락을 고려한 판단 → 더 나은 행동

**4. Actor Head (Policy)**

```python
actor_logits = self.actor_linear(hx)
# hx: (batch_size, 512)
# actor_logits: (batch_size, num_actions)
#
# COMPLEX_MOVEMENT: 12개 행동
# [NOOP, Right, Right+A, Right+B, Right+A+B, A, Left, Left+A, ...]
```

**Logits vs Probabilities**:
```python
# Logits (출력값)
logits = [-2.3, 5.1, 0.8, -1.2, ...]  # 실수 범위

# Probabilities (Softmax 적용 후)
probs = softmax(logits)
# = [0.001, 0.92, 0.03, 0.005, ...]  # 합 = 1.0
# 
# 가장 높은 확률: Right+A (92%)
```

**왜 Logits를 출력하고 Softmax는 나중에?**
```python
# 수치 안정성
# Softmax와 CrossEntropy를 합치면 log 연산 최적화 가능
log_probs = F.log_softmax(logits, dim=1)

# 유연성
# 학습 시: log_softmax 사용
# 추론 시: softmax 또는 argmax 사용
```

**5. Critic Head (Value Function)**

```python
critic_value = self.critic_linear(hx)
# hx: (batch_size, 512)
# critic_value: (batch_size, 1)
#
# 예: tensor([[2.34]]) → 현재 상태의 가치
```

**Value의 의미**:
```python
# 상황 1: 마리오가 깃발 앞
V(s) = 50.0  # 높은 가치 - 곧 클리어

# 상황 2: 마리오가 중간 지점
V(s) = 10.0  # 중간 가치 - 진행 중

# 상황 3: 마리오가 구덩이 앞
V(s) = -20.0  # 낮은 가치 - 위험

# 상황 4: 마리오가 적 앞에서 무적
V(s) = 30.0  # 높은 가치 - 점수 획득 기회
```

**Actor vs Critic**:
```
Actor (정책):
- "무엇을 할까?" 결정
- 출력: 행동 확률
- 예: [Right+Jump: 80%, Right: 15%, ...]

Critic (가치 함수):
- "이 상태가 얼마나 좋은가?" 평가
- 출력: 상태 가치 (단일 숫자)
- 예: V(s) = 15.3
```

**6. Hidden State 반환**

```python
return actor_logits, critic_value, hx, cx
```

**왜 hx, cx를 반환?**
```python
# 에피소드 중간
state_t, _, _, _ = env.reset()
hx, cx = torch.zeros(1, 512), torch.zeros(1, 512)

for t in range(episode_length):
    logits, value, hx, cx = model(state_t, hx, cx)
    # 다음 스텝에서 이전 hx, cx 사용 → 연속성 유지
    action = sample(logits)
    state_t, reward, done, _ = env.step(action)
    
    if done:
        # 에피소드 끝나면 hidden state 리셋
        hx, cx = torch.zeros(1, 512), torch.zeros(1, 512)
        state_t, _, _, _ = env.reset()
```

---

## 2. A2C Loss 계산

### 문제 (`training/train_a2c.py`)

```python
# TODO: A2C 학습을 위한 손실 함수를 계산하세요
# Backward pass through trajectory
for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
    # 힌트 1: GAE (Generalized Advantage Estimation) 계산
    #   - gae = gae * gamma * tau + reward + gamma * next_value - value
    #   - GAE는 advantage를 추정하여 분산을 줄입니다
    
    # 힌트 2: Actor Loss (Policy Gradient)
    #   - actor_loss += log_policy * advantage
    #   - 정책을 advantage 방향으로 업데이트
    
    # 힌트 3: Critic Loss (Value Function)
    #   - R = gamma * R + reward (discounted return)
    #   - critic_loss += (R - value)^2 / 2
    #   - 가치 함수가 실제 return을 예측하도록 학습
    
    # 힌트 4: Entropy Loss (Exploration)
    #   - entropy_loss += entropy
    #   - 탐험을 장려하기 위해 정책의 엔트로피를 증가
    
    #YOUR CODE HERE
    raise NotImplementedError("A2C loss 계산을 구현하세요")

# 힌트 5: Total Loss 계산
# total_loss = -actor_loss + critic_loss - beta * entropy_loss
```

### 정답

```python
# Backward pass through trajectory
for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
    # GAE calculation
    gae = gae * args.gamma * args.tau
    gae = gae + reward + args.gamma * next_value.detach() - value.detach()
    next_value = value

    # Actor loss (policy gradient)
    actor_loss = actor_loss + log_policy * gae

    # Critic loss (value function)
    R = R * args.gamma + reward
    critic_loss = critic_loss + (R - value) ** 2 / 2

    # Entropy loss (exploration bonus)
    entropy_loss = entropy_loss + entropy

# Total loss
total_loss = -actor_loss + critic_loss - args.beta * entropy_loss
```

### 설명

**1. GAE (Generalized Advantage Estimation)**

```python
gae = gae * args.gamma * args.tau
gae = gae + reward + args.gamma * next_value.detach() - value.detach()
next_value = value
```

**GAE 공식**:
```
A^GAE(s_t, a_t) = Σ_{l=0}^∞ (γτ)^l δ_{t+l}

여기서 δ_t = r_t + γV(s_{t+1}) - V(s_t)  # TD error
```

**역방향 계산 ([::-1]을 사용하는 이유)**:
```python
# Trajectory (시간 순서)
# t=0: (v_0, r_0, v_1)
# t=1: (v_1, r_1, v_2)
# t=2: (v_2, r_2, v_3)
# t=3: (v_3, r_3, v_4)

# 역방향으로 계산 (t=3부터 t=0까지)
# t=3: gae_3 = 0 + r_3 + γv_4 - v_3
# t=2: gae_2 = (γτ)gae_3 + r_2 + γv_3 - v_2
# t=1: gae_1 = (γτ)gae_2 + r_1 + γv_2 - v_1
# t=0: gae_0 = (γτ)gae_1 + r_0 + γv_1 - v_0
```

**구체적 예시**:
```python
# 파라미터
gamma = 0.99   # 할인 인수
tau = 1.0      # GAE 파라미터

# Trajectory
# t=3: reward=0, value=0.1, next_value=0
gae_3 = 0 * 0.99 * 1.0
gae_3 += 0 + 0.99 * 0 - 0.1
gae_3 = -0.1  # 부정적 advantage (이 상태는 예상보다 나쁨)

# t=2: reward=10, value=5.0, next_value=0.1
gae_2 = -0.1 * 0.99 * 1.0
gae_2 += 10 + 0.99 * 0.1 - 5.0
gae_2 = -0.099 + 10 + 0.099 - 5.0 = 5.0  # 긍정적 advantage

# t=1: reward=0, value=3.0, next_value=5.0
gae_1 = 5.0 * 0.99 * 1.0
gae_1 += 0 + 0.99 * 5.0 - 3.0
gae_1 = 4.95 + 0 + 4.95 - 3.0 = 6.9  # 매우 긍정적
```

**왜 detach()?**
```python
gae = ... + args.gamma * next_value.detach() - value.detach()
```
- **목적**: Advantage 계산 시 value의 그래디언트 차단
- **이유**: GAE는 고정된 baseline으로 사용, value 자체 학습에는 영향 안 줌
- **효과**: Actor와 Critic 학습 분리

**GAE 파라미터 τ (tau)의 역할**:
```python
# tau = 0: 1-step TD
A_tau=0 = δ_t = r_t + γV(s_{t+1}) - V(s_t)
# 낮은 분산, 높은 bias

# tau = 1: Monte Carlo
A_tau=1 = Σ δ_t = G_t - V(s_t)  # G_t = 실제 return
# 높은 분산, 낮은 bias

# tau = 0.95: 절충안 (일반적)
# 적절한 bias-variance tradeoff
```

**2. Actor Loss (Policy Gradient)**

```python
actor_loss = actor_loss + log_policy * gae
```

**Policy Gradient 이론**:
```
∇J(θ) = E[∇ log π(a|s) * A(s,a)]

"정책의 gradient는
 log 확률의 gradient와
 advantage의 곱의 기댓값"
```

**직관적 이해**:
```python
# Advantage > 0 (좋은 행동)
# → log_policy * positive → positive loss
# → -actor_loss로 gradient 상승
# → 이 행동의 확률 증가

# Advantage < 0 (나쁜 행동)
# → log_policy * negative → negative loss
# → -actor_loss로 gradient 하강
# → 이 행동의 확률 감소
```

**예시**:
```python
# 상황: 적을 밟아서 점수 획득
# action: Right+Jump
# log_policy: log(0.8) = -0.22
# gae: 15.0 (매우 긍정적)
# actor_loss += -0.22 * 15.0 = -3.3

# Total loss에서 -actor_loss
# → -(-3.3) = 3.3 (positive)
# → Gradient 상승 → Right+Jump 확률 증가
```

**3. Critic Loss (Value Function)**

```python
R = R * args.gamma + reward
critic_loss = critic_loss + (R - value) ** 2 / 2
```

**Discounted Return 계산**:
```python
# 역방향 계산
# t=3: R_3 = 0 * 0.99 + r_3 = r_3
# t=2: R_2 = r_3 * 0.99 + r_2 = r_2 + 0.99*r_3
# t=1: R_1 = R_2 * 0.99 + r_1 = r_1 + 0.99*r_2 + 0.99^2*r_3
# t=0: R_0 = R_1 * 0.99 + r_0 = r_0 + 0.99*r_1 + 0.99^2*r_2 + 0.99^3*r_3
```

**MSE Loss**:
```python
critic_loss = (R - value)^2 / 2

# R: 실제 return (Monte Carlo)
# value: 예측한 value (네트워크 출력)
# 목표: value가 R을 잘 예측하도록 학습
```

**예시**:
```python
# t=2
reward = 10  # 적을 밟음
R_prev = 5   # 이전 계산된 R
value = 8    # 네트워크가 예측한 가치

R = 5 * 0.99 + 10 = 14.95
critic_loss += (14.95 - 8)^2 / 2
            = (6.95)^2 / 2
            = 24.15

# Loss가 크다 → value 예측이 부정확
# Backprop → value를 14.95에 가깝게 학습
```

**4. Entropy Loss (Exploration Bonus)**

```python
entropy_loss = entropy_loss + entropy
```

**Entropy 개념**:
```
H(π) = -Σ π(a|s) log π(a|s)

"정책의 불확실성"
```

**예시**:
```python
# 결정적 정책 (낮은 entropy)
probs = [0.99, 0.01, 0, 0, ...]
H = -(0.99*log(0.99) + 0.01*log(0.01))
  ≈ 0.056  # 낮음

# 균등 정책 (높은 entropy)
probs = [0.083, 0.083, 0.083, ...]  # 12개 행동
H = -12 * (0.083 * log(0.083))
  ≈ 2.485  # 높음
```

**왜 Entropy를 최대화?**
```python
# 문제: 정책이 너무 빨리 수렴
# Episode 10: [Right+Jump: 95%, 나머지: 5%]
# → 다른 행동 탐험 안 함
# → 지역 최적해에 갇힘

# 해결: Entropy bonus 추가
# total_loss = ... - beta * entropy
# → Entropy 증가 시 loss 감소
# → 더 균등한 확률 분포 유지
# → 계속 탐험
```

**Beta (엔트로피 계수) 조정**:
```python
beta = 0.0   # Entropy 무시 → 빠른 수렴, 탐험 부족
beta = 0.01  # 약간의 탐험 (일반적)
beta = 0.1   # 많은 탐험 → 느린 수렴, 더 나은 해 발견 가능
```

**5. Total Loss**

```python
total_loss = -actor_loss + critic_loss - args.beta * entropy_loss
```

**부호 설명**:
```python
# Actor Loss: -actor_loss
# → log_policy * advantage를 최대화
# → Gradient 상승

# Critic Loss: +critic_loss
# → (R - V)^2를 최소화
# → Gradient 하강

# Entropy Loss: -beta * entropy_loss
# → Entropy를 최대화
# → 탐험 장려
```

**Loss 균형**:
```python
# 전형적인 값 범위
actor_loss:   -50 ~ 50
critic_loss:  0 ~ 100    (MSE이므로 항상 양수)
entropy_loss: 0 ~ 3      (정책 entropy)

# Beta = 0.01일 때
total_loss = -30 + 45 - 0.01 * 2.0
           = -30 + 45 - 0.02
           = 14.98
```

---

## 핵심 개념 요약

### Actor-Critic의 핵심 아이디어

**Actor (정책)**:
```python
π(a|s) = Policy Network(s)
"이 상태에서 어떤 행동을 할까?"

손실: -log π(a|s) * A(s,a)
"advantage가 양수면 확률 증가, 음수면 확률 감소"
```

**Critic (가치 함수)**:
```python
V(s) = Value Network(s)
"이 상태가 얼마나 좋은가?"

손실: (R - V(s))^2
"실제 return과 예측 value의 차이 최소화"
```

**왜 분리?**
```
Q-Learning:
- Q(s,a)를 직접 학습
- 모든 (s,a) 쌍에 대해 Q-값 저장
- 연속 행동에서 불가능

Actor-Critic:
- π(a|s): 행동 선택 (Actor)
- V(s): 상태 평가 (Critic)
- π와 V를 각각 신경망으로 근사
- 연속 행동에서도 가능
- 더 안정적인 학습
```

### GAE (Generalized Advantage Estimation)

**Advantage의 의미**:
```
A(s,a) = Q(s,a) - V(s)

Q(s,a): 이 행동을 했을 때의 가치
V(s):   평균적인 가치
A(s,a): 이 행동이 평균보다 얼마나 좋은지
```

**GAE의 필요성**:
```
# 문제: Advantage 추정의 분산이 큼
# → 학습 불안정

# Monte Carlo 방식 (높은 분산)
A = R_t - V(s_t)

# TD 방식 (낮은 분산, 높은 bias)
A = r_t + γV(s_{t+1}) - V(s_t)

# GAE (절충안)
A = Σ (γτ)^l δ_{t+l}
# τ로 bias-variance 조절
```

### LSTM의 역할

**왜 LSTM?**
```
단일 프레임으로는 부족:
- 마리오가 점프 중: 올라가는 중? 내려오는 중?
- 적이 가까이: 다가오는 중? 멀어지는 중?
- 블록 앞: 부딪힐 예정? 피할 예정?

LSTM이 기억:
- 이전 몇 프레임의 패턴
- 속도와 방향 정보
- 시간적 맥락
→ 더 정확한 판단
```

**Hidden State 관리**:
```python
# 에피소드 시작
hx, cx = zeros(1, 512), zeros(1, 512)

# 에피소드 진행
for t in range(steps):
    logits, value, hx, cx = model(state, hx, cx)
    # hx, cx가 누적되어 과거 정보 저장

# 에피소드 종료
if done:
    # 리셋! 새 에피소드는 독립적
    hx, cx = zeros(1, 512), zeros(1, 512)
```

---

## 디버깅 가이드

### 자주 발생하는 오류

**1. RuntimeError: LSTM expected batch_first=False**
```python
# 문제
lstm = nn.LSTM(input_size, hidden_size)
# 입력 형태: (seq_len, batch, features)

# 해결: batch_first=True 사용 또는 입력 변환
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
# 입력 형태: (batch, seq_len, features)
```

**2. ValueError: too many values to unpack**
```python
# 문제
for value, log_policy, reward in zip(values, log_policies, rewards):
# 4개를 3개 변수에 언패킹

# 해결
for value, log_policy, reward, entropy in zip(...):
```

**3. RuntimeError: Trying to backward through the graph a second time**
```python
# 문제
gae = gae + reward + gamma * next_value - value
# next_value와 value에 그래디언트 그래프 연결

# 해결
gae = gae + reward + gamma * next_value.detach() - value.detach()
# detach()로 그래디언트 차단
```

**4. Hidden state 차원 불일치**
```python
# 문제
hx = torch.zeros(512)  # (512,)
hx, cx = lstm(x, (hx, cx))  # 오류!

# 해결
hx = torch.zeros(1, 512)  # (batch_size, hidden_size)
cx = torch.zeros(1, 512)
```

**5. 학습이 전혀 안 되는 경우**
```python
# 원인 1: GAE 계산 순서 오류
# 반드시 역방향 ([::-1])

# 원인 2: Total loss 부호 오류
# total_loss = -actor_loss + critic_loss - beta * entropy

# 원인 3: Entropy 계수가 너무 큼
# beta = 0.01 권장 (0.1은 너무 클 수 있음)

# 원인 4: 에피소드 종료 시 hidden state 미리셋
# done일 때 hx, cx = zeros(...)
```

---

## 성능 개선 팁

### 1. 하이퍼파라미터 튜닝

**Learning Rate**
```python
lr = 1e-4   # ✅ 안정적
lr = 1e-3   # ⚠️ 발산 가능
lr = 1e-5   # ⚠️ 너무 느림
```

**GAE 파라미터 (tau)**
```python
tau = 1.0   # ✅ Monte Carlo 방식 (높은 분산)
tau = 0.95  # ✅ 절충안 (권장)
tau = 0.8   # ✅ TD 방식에 가까움 (낮은 분산)
```

**Entropy 계수 (beta)**
```python
beta = 0.01   # ✅ 일반적
beta = 0.001  # ⚠️ 탐험 부족
beta = 0.1    # ⚠️ 너무 많은 탐험, 느린 수렴
```

**Num Local Steps**
```python
num_local_steps = 50   # ✅ 적당한 배치
num_local_steps = 10   # ⚠️ 너무 적음, 불안정
num_local_steps = 200  # ⚠️ 너무 많음, 느린 업데이트
```

### 2. 네트워크 구조 개선

**더 깊은 CNN**
```python
# 기본
Conv1 → Conv2 → Conv3 → Conv4

# 개선: ResNet 블록 추가
Conv1 → ResBlock1 → ResBlock2 → Conv4
```

**더 큰 LSTM**
```python
# 기본
LSTM(1152 → 512)

# 개선: 2-layer LSTM
LSTM(1152 → 512) → LSTM(512 → 512)
```

### 3. 학습 전략

**Reward Shaping**
```python
# 기본
reward = score_diff / 40.0

# 개선: 진행도 보상 추가
reward = score_diff / 40.0 + x_pos_diff / 100.0
```

**Curriculum Learning**
```python
# Stage 1-1부터 시작
world, stage = 1, 1

# 성공률 70% 이상이면 다음 스테이지
if success_rate > 0.7:
    stage += 1
```

**Gradient Clipping**
```python
# 기본
clip_grad_norm_(model.parameters(), 0.5)

# 더 강한 clipping
clip_grad_norm_(model.parameters(), 0.1)
```

---

## 확장 과제

### 1. PPO (Proximal Policy Optimization)
```python
# A2C: 제약 없이 정책 업데이트
loss = -log π(a|s) * A

# PPO: 정책 변화 제한
ratio = π_new(a|s) / π_old(a|s)
loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

### 2. Multi-Environment Training
```python
# A2C: 단일 환경
env = create_env(world, stage)

# A3C: 여러 환경 병렬
envs = [create_env(world, stage) for _ in range(16)]
# 각 환경에서 경험 수집 → 합쳐서 학습
```

### 3. Auxiliary Tasks
```python
# 기본: Actor + Critic만 학습

# 개선: 보조 과제 추가
# - Pixel reconstruction: 다음 프레임 예측
# - Reward prediction: 보상 예측
# → 더 나은 표현 학습
```

---

## 학습 체크리스트

### Forward Pass 검증
- [ ] Conv 레이어 4개 모두 통과
- [ ] ReLU 활성화 함수 적용
- [ ] Flatten 후 LSTM 입력
- [ ] LSTM hidden/cell state 업데이트
- [ ] Actor logits 출력 (num_actions 차원)
- [ ] Critic value 출력 (1 차원)
- [ ] Hidden state 반환

### Loss 계산 검증
- [ ] GAE 계산 (역방향)
- [ ] Actor loss: log_policy * gae
- [ ] Critic loss: (R - V)^2 / 2
- [ ] Entropy loss: entropy 합
- [ ] Total loss: -actor + critic - beta*entropy
- [ ] detach() 사용 (value, next_value)

### 학습 검증
- [ ] Optimizer zero_grad()
- [ ] loss.backward()
- [ ] Gradient clipping
- [ ] Optimizer step()
- [ ] 에피소드 종료 시 hidden state 리셋

---

이 정답 파일은 학습 과정에서 막혔을 때 참고용으로 사용하세요.
가능한 한 스스로 구현해보고, 어려운 부분만 힌트로 활용하는 것을 권장합니다.

