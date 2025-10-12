# 빠른 시작 가이드

## 1. 환경 설정

### 가상환경 생성 및 활성화
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows
```

### 라이브러리 설치
```bash
pip install -r requirements.txt
```

## 2. 기본 실행

### 유전 알고리즘 훈련 (추천)
```bash
python main.py --algorithm genetic --episodes 50 --visualize
```

### 다양한 설정으로 비교
```bash
python main.py --compare --episodes 30 --visualize
```

### 간단한 예제
```bash
python examples/simple_genetic.py
```

### 상호작용 데모
```bash
python examples/interactive_demo.py
```

## 3. 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `--algorithm genetic` | 유전 알고리즘 사용 |
| `--episodes 100` | 에피소드/세대 수 설정 |
| `--visualize` | 결과 시각화 |
| `--test` | 훈련 후 테스트 실행 |
| `--render` | 게임 화면 표시 |
| `--compare` | 다양한 설정으로 성능 비교 |

## 4. 문제 해결

**ModuleNotFoundError**: 가상환경이 활성화되었는지 확인
**느린 훈련**: `--render` 옵션 제거
**그래프 오류**: matplotlib 한글 폰트 설정 확인

행복한 학습 되세요!
