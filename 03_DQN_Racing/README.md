# DQN (Deep Q-Networks) for CarRacing

OpenAI Gymnasium의 CarRacing 환경에서 DQN 알고리즘을 학습하는 프로젝트입니다.

## 목표

이 프로젝트를 통해 다음을 학습합니다:
- **Deep Q-Networks (DQN)**: Q-Learning에 딥러닝을 결합한 알고리즘
- **Experience Replay**: 학습 안정화를 위한 경험 재사용
- **Target Network**: Q-값 추정의 안정성 향상
- **CNN 기반 가치 함수**: 이미지 입력에서 Q-값 추정

## 환경 설정

### Conda 사용 (권장)

```bash
# 1. Conda 환경 생성
conda create -n dqn_racing python=3.12

# 2. 환경 활성화
conda activate dqn_racing

# 3. 패키지 설치
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt

# 4. Box2D 설치 (CarRacing 환경에 필요)
conda install -c conda-forge box2d-py
```

### Pip 사용

```bash
# 1. 가상환경 생성
python -m venv dqn_racing_env

# 2. 환경 활성화
source dqn_racing_env/bin/activate  # macOS/Linux
# 또는
dqn_racing_env\Scripts\activate  # Windows

# 3. 패키지 설치
pip install -r requirements.txt
```

## 디렉토리 구조

```
03_DQN_Racing/
├── training/
│   └── dqn_training.py          # DQN 학습 코드
├── games/
│   ├── test_manual_play.py      # 수동 플레이 테스트
│   └── demo_trained_agent.py    # 학습된 에이전트 시연
├── tutorials/
│   ├── dqn_tutorial.py          # DQN 이론 튜토리얼
│   └── dqn_tutorial.ipynb       # Jupyter 노트북 버전
├── models/
│   └── saved_weights/           # 학습된 모델 저장
├── logs/                         # 학습 로그
└── requirements.txt              # 패키지 의존성
```

## 학습 과제

`training/dqn_training.py` 파일에서 다음 핵심 부분을 구현하세요:

### 1. DQN 네트워크 Forward Pass
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # TODO: CNN을 통과시켜 Q-값 출력
    #YOUR CODE HERE
```

**학습 내용**: CNN으로 이미지에서 특징을 추출하고 Q-값을 예측

### 2. Experience Replay Buffer
```python
def sample(self, batch_size: int) -> Tuple:
    # TODO: 버퍼에서 무작위 배치 샘플링
    #YOUR CODE HERE
```

**학습 내용**: 경험을 재사용하여 학습 안정화 및 샘플 효율성 향상

### 3. Epsilon-Greedy Action Selection
```python
def select_action(self, state, training=True) -> int:
    # TODO: 탐험과 활용의 균형
    #YOUR CODE HERE
```

**학습 내용**: 탐험(exploration)과 활용(exploitation)의 균형

### 4. DQN Update (Bellman Equation)
```python
def update(self) -> Optional[float]:
    # TODO: Q-learning 업데이트 구현
    #YOUR CODE HERE
```

**학습 내용**: Target Network와 Bellman Equation을 이용한 Q-값 업데이트

## 사용 방법

### 1단계: 환경 테스트 (수동 플레이)

```bash
python games/test_manual_play.py
```

**조작법**:
- ↑ (위): 가속
- ↓ (아래): 브레이크
- ← (왼쪽): 좌회전
- → (오른쪽): 우회전
- ESC: 종료
- R: 리셋

### 2단계: DQN 튜토리얼

```bash
# Python 스크립트로 실행
python tutorials/dqn_tutorial.py

# 또는 Jupyter Notebook
jupyter notebook tutorials/dqn_tutorial.ipynb
```

### 3단계: TODO 구현 및 학습

1. `training/dqn_training.py`의 TODO 부분을 구현
2. 학습 실행:

```bash
python training/dqn_training.py --episodes 500
```

**주요 하이퍼파라미터**:
- `--episodes`: 학습 에피소드 수 (기본: 500)
- `--render`: 학습 중 환경 렌더링
- `--load`: 이전 모델에서 이어서 학습

예시:
```bash
# 기본 학습
python training/dqn_training.py

# 1000 에피소드 학습
python training/dqn_training.py --episodes 1000

# 이전 모델에서 이어서 학습
python training/dqn_training.py --load models/saved_weights/dqn_best.pth
```

### 4단계: 학습된 에이전트 시연

```bash
# 기본 시연 (Best 모델 자동 로드)
python games/demo_trained_agent.py

# 특정 모델 시연
python games/demo_trained_agent.py --model models/saved_weights/dqn_best.pth

# 학습된 에이전트 vs 랜덤 에이전트 비교
python games/demo_trained_agent.py --compare --episodes 10

# 인터랙티브 모드
python games/demo_trained_agent.py --interactive
```

## DQN 알고리즘 핵심 개념

### 1. Experience Replay
- **문제**: 연속된 샘플들이 높은 상관관계를 가짐
- **해결**: 과거 경험을 버퍼에 저장하고 무작위로 샘플링
- **효과**: 학습 안정화, 샘플 효율성 향상

### 2. Target Network
- **문제**: Q-값을 추정할 때 같은 네트워크 사용 시 불안정
- **해결**: 별도의 Target Network를 주기적으로 업데이트
- **효과**: Q-값 추정의 안정성 향상

### 3. Bellman Equation
```
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```
- `s`: 현재 상태
- `a`: 선택한 행동
- `r`: 보상
- `s'`: 다음 상태
- `γ`: 할인 인수 (0.99)
- `α`: 학습률 (0.0001)

### 4. CNN Architecture
```
입력: 4 x 84 x 84 (프레임 스택)
    ↓
Conv1: 32 filters, 8x8, stride 4
    ↓
Conv2: 64 filters, 4x4, stride 2
    ↓
Conv3: 64 filters, 3x3, stride 1
    ↓
FC1: 512 units
    ↓
FC2: 4 units (Q-values for each action)
```

## 하이퍼파라미터

```python
HYPERPARAMETERS = {
    'learning_rate': 0.0001,      # Adam 학습률
    'gamma': 0.99,                # 할인 인수
    'epsilon_start': 1.0,         # 초기 탐험률
    'epsilon_end': 0.01,          # 최소 탐험률
    'epsilon_decay': 0.995,       # 탐험률 감쇠
    'batch_size': 32,             # 배치 크기
    'buffer_size': 10000,         # Replay Buffer 크기
    'target_update': 1000,        # Target Network 업데이트 주기
    'frame_stack': 4,             # 프레임 스택 수
    'image_size': (84, 84)        # 입력 이미지 크기
}
```

## 학습 팁

1. **시작은 작게**: 먼저 100-200 에피소드로 코드가 제대로 작동하는지 확인
2. **로그 확인**: 학습 중 epsilon, loss, reward 값을 모니터링
3. **Replay Buffer**: 충분히 채워진 후 (batch_size 이상) 학습 시작
4. **Target Network**: 너무 자주 업데이트하면 불안정, 너무 느리면 학습이 느림
5. **탐험률**: 초기에는 많이 탐험하고, 점차 학습한 정책을 활용

## 성능 벤치마크

- **Random Agent**: -30 ~ -50 (무작위 행동)
- **초기 학습 (100 episodes)**: -20 ~ -10
- **중기 학습 (300 episodes)**: 0 ~ 50
- **완료 학습 (500+ episodes)**: 50 ~ 200+

## 디버깅 가이드

### 학습이 안 되는 경우

1. **Replay Buffer가 비어있음**
   - `batch_size`보다 많은 경험이 쌓여야 학습 시작
   - 초기 몇 에피소드는 버퍼를 채우는 시간

2. **Epsilon이 너무 높음**
   - 무작위 행동만 선택하면 학습 안 됨
   - `epsilon_decay`를 조정하여 더 빨리 감소

3. **Learning Rate가 너무 큼/작음**
   - 너무 크면 불안정, 너무 작으면 느림
   - 0.0001 ~ 0.001 사이 권장

4. **Target Network 업데이트 주기**
   - 너무 자주 업데이트하면 불안정
   - 1000 ~ 10000 스텝 사이 권장

## 추가 학습 자료

- **DQN 원논문**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **Nature DQN**: [Human-level control through deep RL](https://www.nature.com/articles/nature14236)
- **Gymnasium Docs**: [CarRacing Environment](https://gymnasium.farama.org/environments/box2d/car_racing/)

## 확장 과제

1. **Double DQN**: Target Q-값 계산 시 action selection과 evaluation 분리
2. **Dueling DQN**: Value와 Advantage를 분리하여 학습
3. **Prioritized Experience Replay**: 중요한 경험에 더 높은 우선순위
4. **Rainbow DQN**: 여러 DQN 개선 기법을 결합

## 라이선스

이 프로젝트는 교육 목적으로 만들어졌습니다.

## 문의

이슈나 질문이 있으시면 GitHub Issues로 남겨주세요.

