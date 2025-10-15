# Pikachu Volleyball RL Agent

**AlphaGo Zero 스타일의 Self-Play 강화학습으로 학습한 피카츄 배구 AI 에이전트**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

## 프로젝트 개요

이 프로젝트는 고전 웹 게임 "Pikachu Volleyball"을 플레이하는 강화학습 에이전트를 개발합니다. **Proximal Policy Optimization (PPO)** 알고리즘과 **Self-Play** 메커니즘을 사용하여, 에이전트는 자기 자신과 대전하며 점진적으로 실력을 향상시킵니다.

### 핵심 특징

- **Self-Play 학습**: AlphaGo Zero처럼 자기 자신과 대전하며 학습
- **단일 네트워크**: 좌우 대칭 변환을 활용하여 하나의 신경망으로 양쪽 플레이어 제어
- **PPO 알고리즘**: 안정적이고 효율적인 on-policy 강화학습
- **Multi-Discrete Action Space**: x방향(3), y방향(3), 파워히트(2) = 18가지 행동 조합
- **ONNX 배포**: 웹 브라우저, 모바일, 임베디드 시스템에서 실행 가능

## 성능

### 학습 결과 (3M timesteps)

- **평균 득점**: Player 1 = 15.00 (만점), Player 2 = 8.30
- **승률**: 100% (Self-play이지만 결정적 정책 사용 시 선공 우위)
- **게임 길이**: 평균 2,300 프레임 (약 92초)
- **학습 시간**: NVIDIA RTX 5080 기준 약 3시간

### 모델 크기

- **네트워크 파라미터**: 72,201개
- **PyTorch 체크포인트**: 861KB
- **ONNX 모델**: 284KB
- **추론 속도**: ONNX가 PyTorch 대비 2.29배 빠름

## 설치

### 요구사항

- Python 3.12+
- CUDA 11.8+ (GPU 사용 시) 또는 Apple Silicon (MPS)
- 4GB+ RAM
- 1GB+ 디스크 공간

### 설치 과정

```bash
# 저장소 클론
cd /home/ubuntu/project
git clone <repository-url> alphachu
cd alphachu

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 패키지 설치
pip install --upgrade pip

# GPU (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 기타 패키지
pip install -r requirements.txt
```

### requirements.txt

```
gymnasium>=0.29.0
numpy>=1.24.0
scipy>=1.10.0
pygame>=2.5.0
matplotlib>=3.7.0
tensorboard>=2.13.0
onnx>=1.14.0
onnxruntime>=1.15.0
tqdm>=4.65.0
```

## 빠른 시작

### 1. 학습된 모델로 플레이 보기

```bash
# 게임 시각화 (5 에피소드, 결정적 정책)
python scripts/play.py \
  --model models/checkpoints/best_model.pth \
  --episodes 5 \
  --deterministic \
  --scale 2
```

### 2. 모델 평가 (통계)

```bash
# 100 에피소드 평가
python scripts/evaluate_model.py \
  --model models/checkpoints/best_model.pth \
  --episodes 100 \
  --deterministic
```

### 3. 처음부터 학습

```bash
# 짧은 테스트 (5만 스텝, ~5분)
python scripts/train.py --timesteps 50000

# 본격 학습 (100만 스텝, ~2-3시간, GPU 기준)
python scripts/train.py --timesteps 1000000
```

### 4. 학습 재개

```bash
# 체크포인트에서 이어서 학습
python scripts/train.py \
  --resume models/checkpoints/checkpoint_500000.pth \
  --timesteps 1000000
```

### 5. 학습 모니터링

```bash
# Tensorboard 실행 (별도 터미널)
tensorboard --logdir=logs/tensorboard --port=6006

# 브라우저에서 http://localhost:6006 접속
```

## 프로젝트 구조

```
alphachu/
├── README.md                       # 이 파일
├── PPO_SELFPLAY_REPORT.md          # 기술 보고서
├── agent.md                        # 상세 개발 계획
├── WORKFLOW.md                     # 워크플로우 가이드
├── requirements.txt                # Python 패키지
│
├── env/                           # 환경 구현
│   ├── physics.py                 # 물리 엔진 (JavaScript → Python 포팅)
│   ├── pikachu_env.py             # Gymnasium 환경
│   ├── symmetry.py                # 좌우 대칭 변환
│   └── test_env.py                # 환경 검증
│
├── agents/                        # 에이전트 구현
│   ├── network.py                 # Actor-Critic 네트워크
│   ├── ppo.py                     # PPO 알고리즘
│   └── rollout_buffer.py          # Rollout 버퍼 (GAE)
│
├── training/                      # 학습 시스템
│   ├── self_play.py               # Self-Play 트레이너
│   └── config.py                  # 하이퍼파라미터
│
├── evaluation/                    # 평가 및 시각화
│   ├── renderer.py                # Pygame 렌더러
│   └── assets/                    # 게임 그래픽
│
├── deployment/                    # 배포
│   ├── pikachu_agent.onnx         # ONNX 모델 (284KB)
│   ├── export_onnx.py             # ONNX 변환 스크립트
│   ├── test_onnx_inference.py     # ONNX 추론 테스트
│   └── README.md                  # ONNX 사용 가이드
│
├── scripts/                       # 실행 스크립트
│   ├── train.py                   # 학습 실행
│   ├── play.py                    # 게임 시각화
│   ├── evaluate_model.py          # 모델 평가
│   ├── test_env.py                # 환경 테스트
│   └── demo.py                    # 랜덤 플레이 데모
│
├── models/                        # 저장된 모델
│   └── checkpoints/
│       ├── best_model.pth         # Best 모델 (3M steps)
│       └── final_model.pth        # Final 모델
│
└── logs/                          # 학습 로그
    └── tensorboard/               # Tensorboard 로그
```

## 기술 스택

### 강화학습
- **Gymnasium**: 환경 인터페이스
- **PPO**: Proximal Policy Optimization
- **Self-Play**: 자기 대전 학습

### 딥러닝
- **PyTorch**: 신경망 구현
- **Actor-Critic**: 정책(Actor)과 가치(Critic) 동시 학습
- **GAE**: Generalized Advantage Estimation

### 시각화 및 모니터링
- **Pygame**: 게임 렌더링
- **Tensorboard**: 학습 곡선 시각화
- **tqdm**: 진행 상황 표시

### 배포
- **ONNX**: 플랫폼 독립적 모델 포맷
- **ONNX Runtime**: 고속 추론

## 알고리즘 상세

### PPO (Proximal Policy Optimization)

PPO는 OpenAI에서 개발한 on-policy 강화학습 알고리즘으로, 다음과 같은 특징이 있습니다:

1. **Clipped Surrogate Objective**: 정책 업데이트를 제한하여 학습 안정성 확보
2. **Actor-Critic 구조**: 정책과 가치 함수를 동시에 학습
3. **GAE (Generalized Advantage Estimation)**: 분산을 줄인 Advantage 추정
4. **Multiple Epochs**: 수집한 데이터로 여러 번 학습 (on-policy이지만 효율적)

자세한 내용은 [`PPO_SELFPLAY_REPORT.md`](PPO_SELFPLAY_REPORT.md)를 참조하세요.

### Self-Play

AlphaGo Zero에서 영감을 받은 Self-Play 메커니즘:

1. **단일 네트워크**: 하나의 신경망이 양쪽 플레이어 제어
2. **좌우 대칭**: Player 2는 관찰과 행동을 좌우 반전하여 사용
3. **Cold Start**: 랜덤 초기화된 에이전트끼리 대전하며 점진적 개선
4. **Curriculum Learning**: 자연스럽게 난이도가 증가

## 하이퍼파라미터

```python
HYPERPARAMS = {
    # 환경
    "winning_score": 15,
    
    # 네트워크
    "observation_dim": 15,
    "hidden_dims": (256, 256),
    
    # PPO
    "learning_rate": 3e-4,
    "gamma": 0.99,              # 할인율
    "gae_lambda": 0.95,         # GAE 파라미터
    "clip_epsilon": 0.2,        # PPO clip 범위
    "value_coef": 0.5,          # 가치 손실 계수
    "entropy_coef": 0.01,       # 엔트로피 보너스
    "max_grad_norm": 0.5,       # Gradient clipping
    "normalize_advantages": True,
    
    # 학습
    "n_steps": 2048,            # Rollout 길이
    "n_epochs": 10,             # PPO 에포크
    "batch_size": 64,
    
    # 스케줄
    "total_timesteps": 1_000_000,
    "save_freq": 10_000,
    "eval_freq": 10_000,
    "log_freq": 1_000,
}
```

## ONNX 배포

### ONNX 변환

```bash
# PyTorch 모델을 ONNX로 변환
python deployment/export_onnx.py \
  --model models/checkpoints/best_model.pth \
  --output deployment/pikachu_agent.onnx
```

### ONNX 모델 사용 (Python)

```python
import numpy as np
import onnxruntime as ort

# 세션 생성
session = ort.InferenceSession("deployment/pikachu_agent.onnx")

# 관찰 (15차원)
observation = np.random.randn(1, 15).astype(np.float32)

# 추론
x_logits, y_logits, power_logits, value = session.run(
    None, 
    {'observation': observation}
)

# 행동 선택 (Greedy)
x_action = np.argmax(x_logits[0])      # 0=left, 1=stay, 2=right
y_action = np.argmax(y_logits[0])      # 0=stay, 1=jump, 2=down
power_action = np.argmax(power_logits[0])  # 0=no, 1=yes
```

자세한 사용법은 [`deployment/README.md`](deployment/README.md)를 참조하세요.

## 학습 곡선

### Tensorboard 메트릭

- **Rollout/mean_reward_p1, mean_reward_p2**: 평균 보상
- **Rollout/mean_score_p1, mean_score_p2**: 평균 점수 (0-15)
- **Train/policy_loss**: 정책 손실
- **Train/value_loss**: 가치 손실
- **Train/entropy_loss**: 엔트로피 (exploration 지표)
- **Train/approx_kl**: KL divergence (정책 변화량)
- **Train/clip_fraction**: Clipping 비율
- **Eval/mean_score_p1**: 평가 점수
- **Eval/win_rate_p1**: 승률

### 학습 진행 예시

```
Timestep: 100,000
  Mean Score P1: 8.2
  Mean Score P2: 7.5
  Policy Loss: 0.023
  Value Loss: 0.145

Timestep: 500,000
  Mean Score P1: 12.5
  Mean Score P2: 10.8
  Policy Loss: 0.015
  Value Loss: 0.089

Timestep: 1,000,000
  Mean Score P1: 14.8
  Mean Score P2: 9.2
  Policy Loss: 0.008
  Value Loss: 0.042
```

## 문제 해결

### GPU 메모리 부족

```python
# training/config.py 수정
"n_steps": 1024,   # 2048 → 1024
"batch_size": 32,  # 64 → 32
```

### 학습이 불안정한 경우

```python
"clip_epsilon": 0.1,      # 0.2 → 0.1 (더 보수적)
"learning_rate": 1e-4,    # 3e-4 → 1e-4 (더 느리게)
"entropy_coef": 0.02,     # 0.01 → 0.02 (더 많은 exploration)
```

### 학습 속도가 느린 경우

```python
"n_steps": 4096,    # 2048 → 4096 (더 긴 rollout)
"batch_size": 128,  # 64 → 128 (더 큰 배치)
```

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 라이센스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

## 참고 문헌

### 논문

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
- **AlphaGo Zero**: Silver et al., "Mastering the game of Go without human knowledge" (2017)
- **Actor-Critic**: Konda & Tsitsiklis, "Actor-Critic Algorithms" (2000)

### 코드 및 라이브러리

- [Gymnasium](https://gymnasium.farama.org/) - RL 환경 인터페이스
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL 알고리즘 참고
- [Original Pikachu Volleyball](https://gorisanson.github.io/pikachu-volleyball/en/) - 원본 게임
