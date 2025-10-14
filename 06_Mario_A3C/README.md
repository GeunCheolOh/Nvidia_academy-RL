# A2C (Advantage Actor-Critic) for Super Mario Bros

Super Mario Bros 환경에서 A2C 알고리즘을 학습하는 프로젝트입니다.

## 목표

이 프로젝트를 통해 다음을 학습합니다:
- **Actor-Critic**: 정책(Policy)과 가치함수(Value Function)를 동시에 학습
- **Advantage 추정**: GAE를 통한 분산 감소
- **LSTM**: 시간적 의존성을 고려한 순차 의사결정
- **정책 그래디언트**: 연속적인 정책 개선

## 환경 설정

### Pip 사용

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 환경 활성화
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 3. 패키지 설치
pip install -r requirements.txt
```

## 디렉토리 구조

```
06_Mario_A3C/
├── training/
│   ├── train_a2c.py           # A2C 학습 코드 (학생용 - TODO 포함)
│   ├── train_a2c_solution.py  # 정답 코드
│   └── src/
│       ├── model.py           # Actor-Critic 모델 (학생용)
│       ├── model_solution.py  # 정답 모델
│       └── env.py             # 환경 wrapper
├── games/
│   ├── test_manual_play.py    # 수동 플레이 테스트
│   └── demo_trained_agent.py  # 학습된 에이전트 시연
├── tutorials/
│   └── a2c_tutorial.ipynb     # A2C 이론 튜토리얼
├── models/
│   └── saved_weights/         # 학습된 모델 저장
└── logs/                       # 학습 로그
```

## 학습 과제

다음 핵심 부분을 구현하세요:

### 1. Actor-Critic 네트워크 Forward Pass (`training/src/model.py`)

```python
def forward(self, x, hx, cx):
    # TODO: CNN + LSTM + Actor/Critic 헤드 구현
    #YOUR CODE HERE
```

**학습 내용**:
- CNN으로 이미지 특징 추출
- LSTM으로 시간적 패턴 학습
- Actor: 행동 확률 출력
- Critic: 상태 가치 추정

### 2. A2C Loss 계산 (`training/train_a2c.py`)

```python
# TODO: GAE, Actor Loss, Critic Loss, Entropy Loss 계산
#YOUR CODE HERE
```

**학습 내용**:
- **GAE (Generalized Advantage Estimation)**: Advantage 추정으로 분산 감소
- **Actor Loss**: Policy Gradient로 정책 개선
- **Critic Loss**: TD 오차로 가치 함수 학습
- **Entropy Loss**: 탐험 장려

## 사용 방법

### 1단계: 환경 테스트 (수동 플레이)

```bash
python games/test_manual_play.py
```

**조작법**:
- 화살표 키: 이동 (좌/우), 아래 (엎드리기)
- Space: 점프
- A: 달리기/파이어
- Q/ESC: 종료

### 2단계: A2C 튜토리얼

```bash
jupyter notebook tutorials/a2c_tutorial.ipynb
```

### 3단계: TODO 구현 및 학습

1. `training/src/model.py`의 Actor-Critic forward pass 구현
2. `training/train_a2c.py`의 A2C loss 계산 구현
3. 학습 실행:

```bash
python training/train_a2c.py --num-updates 10000
```

**주요 하이퍼파라미터**:
```bash
--world: 월드 번호 (1-8, 기본: 1)
--stage: 스테이지 번호 (1-4, 기본: 1)
--action_type: 행동 공간 (right/simple/complex, 기본: complex)
--lr: 학습률 (기본: 1e-4)
--gamma: 할인 인수 (기본: 0.99)
--num-updates: 업데이트 횟수 (기본: 10000)
```

예시:
```bash
# 기본 학습 (World 1-1)
python training/train_a2c.py

# World 1-2, 20000 업데이트
python training/train_a2c.py --world 1 --stage 2 --num-updates 20000

# 이전 모델에서 이어서 학습
python training/train_a2c.py --load --model models/saved_weights/mario_a3c_best.pth
```

### 4단계: 학습된 에이전트 시연

```bash
# Best 모델 시연
python games/demo_trained_agent.py

# 특정 모델 시연
python games/demo_trained_agent.py --model models/saved_weights/mario_a3c_best.pth

# 다른 스테이지 시연
python games/demo_trained_agent.py --world 1 --stage 2
```

## A2C 알고리즘 핵심 개념

### 1. Actor-Critic 구조

```
         State
           ↓
    [CNN Feature Extraction]
           ↓
      [LSTM Layer]
       ↙        ↘
   Actor      Critic
  (Policy)    (Value)
     ↓           ↓
  Actions   State Value
```

- **Actor**: 정책 π(a|s)를 학습하여 행동 선택
- **Critic**: 가치 함수 V(s)를 학습하여 상태 평가

### 2. Advantage Function

```
A(s, a) = Q(s, a) - V(s)
```

- Q(s, a): 상태 s에서 행동 a를 취했을 때의 가치
- V(s): 상태 s의 평균적인 가치
- A(s, a): 행동 a가 평균보다 얼마나 좋은지

### 3. GAE (Generalized Advantage Estimation)

```
GAE(γ, τ) = Σ (γτ)^t δ_t
여기서 δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

- **γ (gamma)**: 할인 인수 (미래 보상의 중요도)
- **τ (tau)**: GAE 파라미터 (bias-variance tradeoff)
- **목적**: Advantage 추정의 분산을 줄여 안정적 학습

### 4. Loss Functions

```python
# Actor Loss (Policy Gradient)
L_actor = -Σ log π(a|s) * A(s,a)

# Critic Loss (TD Error)
L_critic = Σ (R - V(s))^2

# Entropy Loss (Exploration)
L_entropy = -Σ π(a|s) * log π(a|s)

# Total Loss
L_total = L_actor + L_critic - β * L_entropy
```

## Actor-Critic 네트워크 구조

```python
# CNN Feature Extractor
Conv1: 32 filters, 3x3, stride 2
Conv2: 32 filters, 3x3, stride 2
Conv3: 32 filters, 3x3, stride 2
Conv4: 32 filters, 3x3, stride 2

# LSTM Layer
LSTM: 512 hidden units

# Actor Head
Linear: 512 → num_actions

# Critic Head
Linear: 512 → 1
```

## 하이퍼파라미터

```python
{
    'learning_rate': 1e-4,     # Adam 학습률
    'gamma': 0.99,             # 할인 인수
    'tau': 1.0,                # GAE 파라미터
    'beta': 0.01,              # 엔트로피 계수
    'num_local_steps': 50,     # 업데이트 전 스텝 수
    'num_updates': 10000       # 총 업데이트 횟수
}
```

## 학습 팁

1. **스테이지 선택**: 먼저 World 1-1로 시작하여 알고리즘 검증
2. **업데이트 횟수**: 10000 업데이트면 의미있는 학습 가능
3. **모델 저장**: 100 업데이트마다 자동 저장, best 모델 별도 저장
4. **엔트로피 계수**: beta가 클수록 더 탐험적, 작을수록 더 활용적
5. **LSTM 사용**: 이전 프레임 정보를 기억하여 더 나은 의사결정

## 성능 벤치마크

### World 1-1
- **Random Agent**: 거의 앞으로 못 감
- **초기 학습 (1000 updates)**: 구덩이까지 도달
- **중기 학습 (5000 updates)**: 중간 지점 도달
- **완료 학습 (10000+ updates)**: 깃발 도달 가능

### 성공 지표
- **x_pos > 3000**: 깃발 도달
- **flag_get = True**: 스테이지 클리어

## 디버깅 가이드

### 학습이 안 되는 경우

1. **보상이 계속 음수**
   - 커스텀 보상 함수 확인 (`env.py`)
   - 진행도 보상이 제대로 계산되는지 확인

2. **모델이 한 방향으로만 이동**
   - 엔트로피 계수(beta) 증가
   - 학습률 조정

3. **Loss가 발산**
   - Gradient clipping 확인 (0.5로 설정됨)
   - 학습률 감소

4. **LSTM hidden state 오류**
   - 에피소드 종료 시 hidden state 리셋 확인
   - Detach 사용 확인

## Action Space

### RIGHT_ONLY (5 actions)
- 0: NOOP
- 1: Right
- 2: Right + A (Run)
- 3: Right + B (Jump)
- 4: Right + A + B (Run + Jump)

### SIMPLE_MOVEMENT (7 actions)
- RIGHT_ONLY + Left, A

### COMPLEX_MOVEMENT (12 actions)
- SIMPLE_MOVEMENT + Down, Up, Left combinations

## 추가 학습 자료

- **A3C 원논문**: [Asynchronous Methods for Deep RL](https://arxiv.org/abs/1602.01783)
- **GAE 논문**: [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438)
- **Super Mario Bros Gym**: [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

## 확장 과제

1. **PPO (Proximal Policy Optimization)**: 더 안정적인 정책 업데이트
2. **멀티 스테이지 학습**: 여러 스테이지를 번갈아가며 학습
3. **Curriculum Learning**: 쉬운 스테이지부터 어려운 스테이지로
4. **Curiosity-driven Exploration**: 내재적 동기부여 추가

## 라이선스

이 프로젝트는 교육 목적으로 만들어졌습니다.

## 문의

이슈나 질문이 있으시면 GitHub Issues로 남겨주세요.

