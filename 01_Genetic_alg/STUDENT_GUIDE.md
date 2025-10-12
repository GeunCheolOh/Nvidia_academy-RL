# 학생용 실습 가이드

이 프로젝트는 유전 알고리즘을 학습하기 위한 교육용 자료입니다.

## 시작하기 전에

### 필수 요구사항
- Python 3.7 이상
- 기본적인 Python 프로그래밍 지식
- NumPy 라이브러리에 대한 기본 이해

### 권장 사항
- Jupyter Notebook 사용 경험
- 신경망에 대한 기본 개념
- 최적화 알고리즘에 대한 이해

## 설치 가이드

### 1단계: 저장소 클론

```bash
git clone <repository-url>
cd 1_1_Genetic_alg
```

### 2단계: 가상환경 설정

**자동 설치 (권장)**
```bash
chmod +x setup_env.sh
./setup_env.sh
```

**수동 설치**
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 라이브러리 설치
pip install -r requirements.txt
```

### 3단계: 설치 확인

```bash
python main.py
```

정상적으로 설치되었다면 프로젝트 소개 메시지가 표시됩니다.

## 학습 경로

### 초급: Jupyter 노트북으로 시작 (추천)

1. Jupyter Notebook 실행
```bash
jupyter notebook
```

2. 노트북을 순서대로 학습:
   - `notebooks/01_snake_game_basics.ipynb` - 뱀게임 이해하기
   - `notebooks/02_genetic_algorithm_basics.ipynb` - 유전 알고리즘 기초
   - `notebooks/03_snake_genetic_integration.ipynb` - 게임과 알고리즘 결합
   - `notebooks/04_advanced_techniques.ipynb` - 고급 기법
   - `notebooks/05_real_world_applications.ipynb` - 실제 응용

### 중급: 스크립트로 실습

1. **간단한 훈련 실행**
```bash
python train.py --generations 30 --population_size 20 --visualize
```

2. **훈련된 모델 테스트**
```bash
python play.py --model models/snake_ga_best.npz --episodes 3
```

3. **파라미터 실험**
```bash
# 큰 개체군
python train.py --population_size 100 --generations 50

# 높은 돌연변이율
python train.py --mutation_rate 0.2 --generations 50

# 다른 선택 방법
python train.py --selection_method roulette --generations 50
```

### 고급: 코드 수정 및 확장

1. **피트니스 함수 수정** (`algorithms/genetic.py`)
   - 다른 보상 체계 시도
   - 페널티 조정

2. **신경망 구조 변경** (`algorithms/genetic.py`)
   - 은닉층 개수 조정
   - 활성화 함수 변경

3. **새로운 선택 방법 추가** (`algorithms/genetic.py`)
   - 자신만의 선택 알고리즘 구현

## 실습 과제

### 과제 1: 파라미터 최적화
다음 파라미터를 조정하며 최적의 조합을 찾으세요:
- 개체군 크기: 20, 50, 100
- 돌연변이율: 0.05, 0.1, 0.2
- 교차율: 0.6, 0.8, 0.9

**목표**: 가장 빠르게 좋은 성능에 도달하는 설정 찾기

### 과제 2: 선택 방법 비교
세 가지 선택 방법의 성능을 비교하세요:
- Tournament Selection
- Roulette Wheel Selection
- Rank-based Selection

**목표**: 각 방법의 장단점 분석하기

### 과제 3: 신경망 구조 실험
다양한 신경망 구조를 시도하세요:
- 얕은 네트워크: `--hidden_layers 32`
- 중간 네트워크: `--hidden_layers 64 32`
- 깊은 네트워크: `--hidden_layers 128 64 32`

**목표**: 네트워크 깊이가 성능에 미치는 영향 이해하기

### 과제 4: 피트니스 함수 개선
`algorithms/genetic.py`에서 피트니스 함수를 수정하세요:
- 현재: `fitness = avg_score * 100 + avg_length * 0.1`
- 실험: 다른 가중치나 추가 메트릭 사용

**목표**: 더 효율적인 학습을 위한 피트니스 함수 설계

## 일반적인 문제 해결

### 문제 1: Pygame 설치 오류
```bash
# macOS
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
pip install pygame

# Ubuntu/Debian
sudo apt-get install python3-pygame

# 또는 텍스트 모드 사용
python play.py --model models/snake_ga_best.npz --text_mode
```

### 문제 2: 훈련이 너무 느림
- 개체군 크기를 줄이세요: `--population_size 20`
- 세대 수를 줄이세요: `--generations 30`
- 시각화를 끄세요: `--visualize` 플래그 제거

### 문제 3: 성능이 향상되지 않음
- 더 오래 훈련하세요: `--generations 200`
- 개체군 크기를 늘리세요: `--population_size 100`
- 다른 선택 방법을 시도하세요
- 학습률(돌연변이율)을 조정하세요

### 문제 4: 메모리 부족
- 개체군 크기 감소
- 보드 크기 감소: `--board_width 10 --board_height 10`
- 평가 에피소드 수 감소 (코드 수정 필요)

## 평가 기준

### 기본 이해도 (40점)
- [ ] 유전 알고리즘의 기본 원리 설명 가능
- [ ] 선택, 교차, 돌연변이 개념 이해
- [ ] 신경망의 역할 이해
- [ ] 코드 구조 파악

### 실습 능력 (40점)
- [ ] 기본 훈련 스크립트 실행 성공
- [ ] 파라미터 조정하며 실험
- [ ] 결과 분석 및 해석
- [ ] 문제 해결 능력

### 응용 능력 (20점)
- [ ] 코드 수정 및 개선
- [ ] 새로운 아이디어 구현
- [ ] 창의적인 실험 설계
- [ ] 결과 문서화

## 추가 학습 자료

### 온라인 강의
- [Genetic Algorithms - MIT OpenCourseWare](https://ocw.mit.edu/)
- [Deep Learning Specialization - Coursera](https://www.coursera.org/)

### 논문
- Holland, J. H. (1992). "Genetic Algorithms"
- Goldberg, D. E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning"

### 책
- "Introduction to Evolutionary Computing" by Eiben & Smith
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron

## 도움 받기

### 질문하기 전에
1. 오류 메시지를 자세히 읽어보세요
2. 구글에서 오류 메시지로 검색해보세요
3. 코드를 단계별로 디버깅해보세요
4. 관련 문서를 다시 읽어보세요

### 질문할 때 포함할 내용
- 실행한 정확한 명령어
- 전체 오류 메시지
- Python 버전 및 OS 정보
- 이미 시도해본 해결 방법

## 프로젝트 제출

### 제출 파일
1. 수정한 코드 (있는 경우)
2. 실험 결과 (그래프, 로그)
3. 실험 보고서 (분석 및 결론)
4. README (수정 사항 및 실험 방법 설명)

### 보고서 구성
1. **서론**: 실험 목적
2. **방법**: 사용한 파라미터 및 설정
3. **결과**: 그래프 및 통계
4. **분석**: 결과 해석
5. **결론**: 배운 점 및 개선 방안

---

**Good Luck!**

