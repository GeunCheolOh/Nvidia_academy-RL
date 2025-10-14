# DQN Racing Environment Setup Guide

이 프로젝트는 **두 가지 설치 방법**을 제공합니다. 선호하는 방법을 선택하세요.

## 🐍 방법 1: Conda 설치 (권장 - OS 독립적)

### ✅ 장점
- **모든 OS에서 동일한 명령어**
- **Box2D 자동 설치** (컴파일 불필요)
- **의존성 충돌 최소화**

### 1단계: Conda 설치
```bash
# macOS
brew install miniconda

# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows: https://docs.conda.io/en/latest/miniconda.html
```

### 2단계: 환경 생성
```bash
cd 2_1_DQN_Racing
python setup/setup_conda_env.py
```

### 3단계: 환경 활성화
```bash
# 방법 1
conda activate dqn_racing_conda

# 방법 2 (편의 스크립트)
./activate_conda_env.sh      # macOS/Linux
activate_conda_env.bat       # Windows
```

---

## 🐍 방법 2: Pip 설치 (기존 방식)

### ⚠️ 제약사항
- **OS별 다른 설치 과정** (Box2D 때문에)
- **컴파일 도구 필요할 수 있음**

### 1단계: 환경 설정
```bash
cd 2_1_DQN_Racing
python setup/setup_local_env.py
```

### 2단계: Box2D 수동 설치 (필요시)
```bash
# macOS
brew install swig
pip install 'gymnasium[box2d]'

# Linux
sudo apt-get install swig build-essential python3-dev
pip install 'gymnasium[box2d]'

# Windows
# Visual Studio Build Tools 설치 후
pip install 'gymnasium[box2d]'
```

### 3단계: 환경 활성화
```bash
source dqn_racing_env/bin/activate  # macOS/Linux
dqn_racing_env\\Scripts\\activate     # Windows
```

---

## 🎮 사용법

환경이 활성화된 후:

```bash
# 수동 게임 플레이
python games/test_manual_play.py

# DQN 튜토리얼
python tutorials/dqn_tutorial.py

# AI 학습 시작
python training/dqn_training.py

# 학습된 AI 시연
python games/demo_trained_agent.py
```

## 🎯 게임 조작법

- **↑** (위): 가속
- **↓** (아래): 브레이크  
- **←** (왼쪽): 좌회전
- **→** (오른쪽): 우회전
- **ESC**: 종료
- **R**: 리셋
- **SPACE**: 일시정지

## 🔧 문제 해결

### CarRacing 환경이 안될 때
```bash
# Conda 사용자
conda install -c conda-forge box2d-py --force-reinstall

# Pip 사용자  
pip install 'gymnasium[box2d]' --force-reinstall
```

### 의존성 충돌 시
```bash
# 환경 재생성
rm -rf dqn_racing_env  # pip 방식
conda env remove -n dqn_racing_conda  # conda 방식

# 다시 설치
python setup/setup_local_env.py     # pip
python setup/setup_conda_env.py     # conda
```

## 📊 성능 비교

| 항목 | Conda | Pip |
|------|-------|-----|
| OS 독립성 | ✅ 우수 | ❌ 제한적 |
| 설치 편의성 | ✅ 간단 | ⚠️ 복잡 |
| Box2D 설치 | ✅ 자동 | ❌ 수동 |
| 의존성 관리 | ✅ 우수 | ⚠️ 보통 |
| 설치 시간 | ⚠️ 느림 | ✅ 빠름 |

## 📚 추가 자료

- [DQN 논문](https://arxiv.org/abs/1312.5602)
- [Gymnasium 문서](https://gymnasium.farama.org/)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)

---

**권장**: 처음 사용자는 **Conda 방식**을, 숙련된 사용자는 **Pip 방식**을 사용하세요.