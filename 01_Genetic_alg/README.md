# Snake Genetic Algorithm - 학생 실습 프로젝트

유전 알고리즘(Genetic Algorithm)으로 뱀게임 AI를 학습시키는 교육용 프로젝트입니다.
**학생들은 `train_ga.py`의 TODO를 완성하여 유전 알고리즘의 핵심 원리를 직접 구현합니다.**

## 프로젝트 특징

- **센서 기반 입력** (6차원): 전체 보드 대신 효율적인 센서 데이터 사용
- **빠른 학습**: 10세대 이내에 점수 10+ 달성
- **실습 중심**: 4가지 TODO로 핵심 알고리즘 직접 구현

---

## 빠른 시작

### 1. 환경 설정

**사전 요구사항:** Miniconda 또는 Anaconda 설치 필요
- Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Anaconda: https://www.anaconda.com/download

```bash
# 자동 설치 (권장)
chmod +x setup_env.sh
./setup_env.sh

# 수동 설치
conda create -n snake_ga python=3.9 -y
conda activate snake_ga
pip install -r requirements.txt
```

### 2. 매뉴얼 플레이
```bash
conda activate snake_ga  # 환경 활성화
python manual_play.py
```

### 3. TODO 구현 후 학습 실행
```bash
conda activate snake_ga  # 환경 활성화

# 빠른 테스트 (10세대)
python train_ga.py --population 20 --generations 10

# 시각화 보면서 확인
python train_ga.py --population 20 --generations 10 --render
```

### 4. 학습된 모델 테스트
```bash
python test_agent.py --model models/snake_ga_best.npz
```

---

## 실습 과제: TODO 10개 완성하기

### Part 1: 신경망 구현 (`genome.py`)

#### TODO 1-4: 순전파 (Forward Propagation)

**위치:** `genome.py` 32-54번 줄

**목표:** 신경망의 각 층을 통과하는 순전파 구현

```python
# 각 층마다 같은 패턴 반복
net = np.matmul(net, self.w?)  # 행렬 곱
net = self.relu(net)           # 활성화 함수 (출력층만 softmax)
```

**신경망 구조:**
```
입력(6) -> w1 -> (10) -> w2 -> (20) -> w3 -> (10) -> w4 -> 출력(3)
```

#### TODO 5: ReLU 활성화 함수

**위치:** `genome.py` 66-71번 줄

```python
# 음수는 0으로, 양수는 그대로
return x * (x >= 0)  # 또는 np.maximum(0, x)
```

#### TODO 6: Softmax 활성화 함수

**위치:** `genome.py` 84-90번 줄

```python
# 출력을 확률 분포로 변환
exp_x = np.exp(x - np.max(x))
return exp_x / np.sum(exp_x)
```

---

### Part 2: 유전 알고리즘 구현 (`train_ga.py`)

### TODO 7: 적합도 평가 (Fitness Evaluation)

**위치:** `train_ga.py` 120-124번 줄

```python
genome.fitness = ???  # fitness 값을 저장
```

---

### TODO 8: 선택 (Selection)

**위치:** `train_ga.py` 143-147번 줄

```python
genomes.sort(key=lambda x: x.???, reverse=???)
```

---

### TODO 9: 교차 (Crossover)

**위치:** `train_ga.py` 192-222번 줄

```python
# w1, w2, w3, w4 각각 교차
cut = random.randint(0, new_genome.w1.shape[1])
new_genome.w1[:, :cut] = ???.w1[:, :cut]
new_genome.w1[:, cut:] = ???.w1[:, cut:]
```

---

### TODO 10: 돌연변이 (Mutation)

**위치:** `train_ga.py` 224-256번 줄

```python
# w1, w2, w3, w4 각각 돌연변이
new_genome.w1 += new_genome.w1 * noise * random_sign
```

---

## 실행 옵션

### train_ga.py
```bash
--population 50          # 개체군 크기
--generations 100        # 최대 세대 수
--mutation_prob 0.4     # 돌연변이 확률
--render                # 시각화 활성화 (느림)
--batch_size 10         # 진행 상황 표시 주기
```

### 학습 팁
- **빠른 학습**: 시각화 끄기 (기본값), `--batch_size 10`
- **학습 확인**: `--render` 옵션으로 실시간 관찰
- **성능 향상**: `--population 50~100`, `--generations 100+`

---

## 문제 해결

### NotImplementedError 발생
- TODO 부분을 구현하지 않았거나 `raise NotImplementedError` 줄을 삭제하지 않음
- 힌트를 참고하여 코드 작성 후 해당 줄 삭제

### 학습이 진행되지 않음
- TODO 1: fitness가 제대로 저장되었는지 확인
- TODO 2: reverse=True로 내림차순 정렬 확인
- TODO 3, 4: w1, w2, w3, w4 모두 구현했는지 확인

### 학습이 너무 느림
```bash
# 시각화 끄기 (5-10배 빠름)
python train_ga.py --population 50 --generations 100

# 개체군 크기 감소
python train_ga.py --population 20 --generations 50
```

---

## 기대 결과

- **5세대**: fitness 50~100, score 3~5
- **10세대**: fitness 100~200, score 5~10
- **30세대**: fitness 200+, score 10~15

---

## 체크리스트

**genome.py:**
- [ ] TODO 1-4: 순전파 4개 층 구현
- [ ] TODO 5: ReLU 함수
- [ ] TODO 6: Softmax 함수

**train_ga.py:**
- [ ] TODO 7: 적합도 저장
- [ ] TODO 8: 정렬
- [ ] TODO 9: w1, w2, w3, w4 교차
- [ ] TODO 10: w1, w2, w3, w4 돌연변이

**최종:**
- [ ] 모든 `raise NotImplementedError` 삭제
- [ ] 학습 실행 및 결과 확인

---

## 생성되는 파일

- `models/snake_ga_genN.npz` - N세대 모델
- `models/snake_ga_best.npz` - 최종 최고 모델
- `logs/training_log_*.json` - 학습 로그
- `logs/training_plot_*.png` - 학습 진행 그래프

---

## 유전 알고리즘 개념

1. **개체군(Population)**: 여러 해 후보들 (신경망들)
2. **적합도(Fitness)**: 각 해의 품질 (게임 점수)
3. **선택(Selection)**: 우수한 개체 골라내기
4. **교차(Crossover)**: 부모 유전자 조합
5. **돌연변이(Mutation)**: 무작위 변화로 다양성 확보

이 5단계가 반복되면서 점점 더 나은 AI가 진화합니다!

---

## 정답 확인

모든 TODO를 완성한 후 정답과 비교:
```bash
# 신경망 구현 확인
diff genome.py genome_solution.py

# 유전 알고리즘 확인
diff train_ga.py train_ga_solution.py
```

**주의**: 학습 전에 스스로 해결해보세요!

---

## 프로젝트 구조

```
01_Genetic_alg/
├── snake_game.py          # 뱀 게임 환경 (센서 입력, 보상 시스템)
├── genome.py              # 신경망 구조 (TODO 구현 필요)
├── genome_solution.py     # 신경망 정답 버전
├── train_ga.py           # 학습 스크립트 (TODO 구현 필요)
├── train_ga_solution.py  # 학습 정답 버전
├── manual_play.py        # 키보드로 직접 플레이
├── test_agent.py         # 학습된 모델 테스트
├── setup_env.sh          # Conda 환경 자동 설정
└── requirements.txt      # 의존성 패키지
```

---

질문이 있으면 강사에게 문의하세요!
