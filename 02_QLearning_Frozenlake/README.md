# FrozenLake Q-Learning 실습

Q-Learning 알고리즘을 사용하여 FrozenLake 환경에서 강화학습을 구현하는 실습 프로젝트입니다.

## 프로젝트 개요

- **목표**: 벨만 방정식과 Q-러닝의 핵심 개념을 구현 중심으로 체득
- **환경**: OpenAI Gymnasium FrozenLake-v1
- **알고리즘**: Q-Learning with ε-greedy exploration
- **성능 목표**: 4x4 FrozenLake에서 70% 이상 성공률 달성

## 학습 과제

프로젝트의 핵심 스크립트들에는 구현해야 할 부분이 `#YOUR CODE HERE`로 가려져 있습니다.
각 파일을 완성하면서 Gymnasium 환경 사용법과 Q-Learning 알고리즘을 배울 수 있습니다.

### 1. Gymnasium 환경 이해 (`frozenlake_keyboard_agent.py`)
키보드로 직접 플레이하면서 Gymnasium 환경의 기본 인터페이스를 학습합니다:
- `gym.make()` - 환경 생성
- `env.reset()` - 환경 초기화 및 초기 상태 받기
- `env.step(action)` - 행동 실행 및 결과 받기 (state, reward, terminated, truncated, info)
- `env.render()` - 화면 렌더링

### 2. Q-Learning 훈련 (`q_learning_train.py`)
Q-Learning 알고리즘의 핵심을 직접 구현합니다:
1. **Q-table 초기화** - numpy zeros 배열 생성
2. **ε-greedy 정책** - 탐험과 활용의 균형
3. **Q-learning 업데이트** - 벨만 방정식 구현
4. **Epsilon decay** - 탐험률 점진적 감소

### 3. 학습된 모델 평가 (`q_learning_eval.py`)
학습된 Q-table을 로드하고 평가하는 방법을 학습합니다:
- Q-table 로드 (`load_q_table`)
- Greedy 정책 구현 (탐험 없이 최선의 행동만 선택)
- 환경과의 상호작용

각 부분에는 상세한 힌트가 주석으로 제공됩니다.

## 프로젝트 구조

```
atari/
├── notebooks/
│   ├── 01_Qlearning_Theory_Workflow.ipynb    # 이론 설명 및 워크플로우
│   └── 02_Qlearning_FrozenLake_Training.ipynb # 실습 및 훈련
├── scripts/
│   ├── frozenlake_keyboard_agent.py          # 키보드 조작 에이전트
│   ├── q_learning_train.py                   # Q-Learning 훈련 스크립트
│   └── q_learning_eval.py                    # 모델 평가 스크립트
├── weights/
│   └── (훈련된 Q-table 저장 위치)
├── utils/
│   ├── plotting.py                           # 시각화 함수
│   ├── io.py                                 # 입출력 함수
│   └── __init__.py
├── venv/                                     # Python 가상환경
├── CLAUDE.md                                 # 프로젝트 규칙
├── README.md
└── requirements.txt
```

## 설치 및 실행

### 1. 환경 설정

#### 방법 1: Conda 환경 (권장)

```bash
# Conda 환경 생성 및 활성화
conda create -n frozenlake python=3.10 -y
conda activate frozenlake

# 의존성 설치
pip install -r requirements.txt
```

#### 방법 2: Python venv

```bash
# Python 가상환경 생성 및 활성화
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 키보드 조작 체험

먼저 FrozenLake 환경을 직접 체험해보세요:

```bash
# 정확한 조작을 위해 슬리피 모드 OFF (기본값)
python scripts/frozenlake_keyboard_agent.py --map 4x4

# 슬리피 모드 ON (확률적 이동)
python scripts/frozenlake_keyboard_agent.py --map 4x4 --slippery
```

**조작법:**
- 화살표 키: 이동 (↑↓←→)
- Space: 에피소드 리셋
- R: 통계 보기
- Q: 종료

**참고:** 슬리피 모드에서는 의도한 방향으로 정확히 가지 않고 미끄러집니다 (각 방향 1/3 확률)

### 3. Q-Learning 훈련

```bash
# 기본 설정으로 훈련 (4x4 고정 맵)
python scripts/q_learning_train.py --episodes 5000
# 저장 위치: weights/q_table_4x4.npy

# 8x8 맵으로 훈련
python scripts/q_learning_train.py --episodes 10000 --map 8x8
# 저장 위치: weights/q_table_8x8.npy

# 랜덤 맵으로 훈련 (일반화 성능 향상)
python scripts/q_learning_train.py --episodes 5000 --random-map
# 저장 위치: weights/q_table_4x4_random.npy

# 하이퍼파라미터 조정
python scripts/q_learning_train.py \
    --episodes 5000 \
    --alpha 0.1 \
    --gamma 0.95 \
    --eps-decay 0.995 \
    --map 4x4 \
    --slippery \
    --random-map
```

**저장 파일 형식:**
- 4x4 고정: `q_table_4x4.npy`
- 4x4 랜덤: `q_table_4x4_random.npy`
- 8x8 고정: `q_table_8x8.npy`
- 8x8 랜덤: `q_table_8x8_random.npy`

**고정 맵 vs 랜덤 맵:**
- **고정 맵** (기본값): 동일한 맵에서 반복 학습, 특정 맵 최적화
- **랜덤 맵** (`--random-map`): 매 실행마다 다른 맵 생성, 일반화된 정책 학습

### 4. 학습된 모델 평가

```bash
# 4x4 모델 성능 평가
python scripts/q_learning_eval.py --episodes 100
# 기본값: weights/q_table_4x4.npy 로드

# 8x8 모델 평가
python scripts/q_learning_eval.py --episodes 100 --load-path weights/q_table_8x8.npy --map 8x8

# 랜덤 맵 모델 평가
python scripts/q_learning_eval.py --episodes 100 --load-path weights/q_table_4x4_random.npy

# 시각적 데모 (3 에피소드 천천히 시연)
python scripts/q_learning_eval.py --demonstrate

# 자세한 Q-값 분석과 함께 시연
python scripts/q_learning_eval.py --demonstrate --watch-mode --demo-episodes 5
```

## 노트북 실습

### 1. 이론 워크플로우 (`01_Qlearning_Theory_Workflow.ipynb`)

- 벨만 방정식과 Q-Learning 이론
- ε-greedy 탐험 전략
- 하이퍼파라미터의 역할
- 평가 지표 설명

### 2. 실습 및 훈련 (`02_Qlearning_FrozenLake_Training.ipynb`)

- Q-Learning 알고리즘 구현
- 실제 훈련 및 성능 분석
- 하이퍼파라미터 실험
- 결과 시각화 및 모델 저장

## 하이퍼파라미터 가이드

### 주요 하이퍼파라미터

- **학습률 (α)**: 0.05~0.3 (권장: 0.1)
  - 높을수록 빠른 학습, 낮을수록 안정적 수렴
- **할인인수 (γ)**: 0.9~0.99 (권장: 0.95)
  - 미래 보상의 중요도
- **ε 감쇠율**: 0.99~0.999 (권장: 0.995)
  - 탐험률 감소 속도

### 성능별 설정 예시

**빠른 학습 (불안정할 수 있음):**
```bash
--alpha 0.3 --gamma 0.95 --eps-decay 0.99
```

**안정적 학습 (느림):**
```bash
--alpha 0.05 --gamma 0.99 --eps-decay 0.998
```

**균형 잡힌 설정 (권장):**
```bash
--alpha 0.1 --gamma 0.95 --eps-decay 0.995
```

## 환경별 실행 가이드

### 로컬 환경 (Windows/macOS/Linux)

1. 위의 설치 과정 따라 실행
2. pygame 창에서 시각적 렌더링 가능
3. 키보드 조작 지원


## 성능 벤치마크

### FrozenLake 4x4 (is_slippery=True)

- **무작위 정책**: ~5% 성공률
- **목표 성능**: 70% 이상 성공률

### FrozenLake 8x8

- 더 큰 상태 공간으로 인해 학습 시간 증가
- 1,000,000+ 에피소드 훈련 권장

## 문제 해결

### 일반적인 문제

1. **낮은 성공률 (<50%)**
   - 훈련 에피소드 수 증가
   - 학습률 조정 (0.1~0.3)
   - ε 감쇠율 조정 (더 느리게)

2. **불안정한 학습**
   - 학습률 감소 (0.05~0.1)
   - 여러 번 실험 후 평균 성능 확인

3. **pygame 오류**
   - 헤드리스 모드 사용: `--headless`
   - 또는 render_mode를 "rgb_array"로 설정


## 실험 아이디어

1. **다양한 맵 크기 비교** (4x4 vs 8x8)
2. **슬리피 vs 논슬리피 환경** 성능 차이
3. **고정 맵 vs 랜덤 맵** 일반화 성능 비교
4. **하이퍼파라미터 그리드 서치**
5. **학습 곡선 분석** (수렴 속도)
6. **정책 시각화** (학습된 경로 분석)

## 확장 과제

- [ ] Double Q-Learning 구현
- [ ] SARSA 알고리즘과 비교
- [ ] 함수 근사 (Q-Network) 구현
- [ ] 다른 Gymnasium 환경 적용
- [ ] 하이퍼파라미터 자동 최적화

## 참고 자료

- [Sutton & Barto: Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- [Q-Learning 원논문: Watkins & Dayan (1992)](https://link.springer.com/article/10.1007/BF00992698)

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.