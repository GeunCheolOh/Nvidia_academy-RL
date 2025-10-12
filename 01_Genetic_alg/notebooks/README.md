# 주피터 노트북 학습 가이드

뱀게임부터 유전 알고리즘까지의 완전한 학습 과정을 단계별로 정리한 노트북 시리즈입니다.

## 노트북 구성

### 1. [01_snake_game_basics.ipynb](01_snake_game_basics.ipynb)
**뱀게임 기초 구현**
- 게임 규칙과 구조 이해
- 파이썬으로 뱀게임 구현
- 상태 표현과 행동 정의
- 보상 시스템 설계
- 랜덤 플레이어 테스트

### 2. [02_genetic_algorithm_basics.ipynb](02_genetic_algorithm_basics.ipynb)
**유전 알고리즘 기본 원리**
- 다윈의 진화론과 유전 알고리즘
- 선택, 교차, 돌연변이 구현
- 간단한 최적화 문제로 실습
- 진화 과정 시각화
- 파라미터 영향 분석

### 3. [03_snake_genetic_integration.ipynb](03_snake_genetic_integration.ipynb)
**뱀게임과 유전 알고리즘 결합**
- 신경망 구현
- 게임 상태를 신경망 입력으로 변환
- 유전 알고리즘으로 신경망 훈련
- 학습 과정 시각화
- AI 성능 평가

### 4. [04_advanced_techniques.ipynb](04_advanced_techniques.ipynb)
**고급 기법 및 최적화**
- 다양한 선택 방법 비교
- 교차 전략 비교
- 적응적 돌연변이
- 엘리트 전략과 다양성 보존
- 하이퍼파라미터 최적화

### 5. [05_real_world_applications.ipynb](05_real_world_applications.ipynb)
**실제 응용 및 프로젝트**
- 완전한 훈련 파이프라인
- 다른 게임에 적용 방법
- 실제 최적화 문제들
- 성능 향상 기법들
- 확장 프로젝트 아이디어

## 사용 방법

### 1. 순서대로 학습
노트북은 순서대로 학습하도록 설계되었습니다:
```
01 → 02 → 03 → 04 → 05
```

### 2. 실행 환경 준비
```bash
# 주피터 노트북 설치
pip install jupyter notebook

# 프로젝트 디렉토리에서 실행
cd 1_1_Genetic_alg
jupyter notebook
```

### 3. 각 노트북 실행
- 셀을 순서대로 실행
- 코드를 수정하며 실험
- 결과를 관찰하고 분석

## 학습 목표

### 이론적 이해
- [ ] 강화학습의 기본 개념
- [ ] 유전 알고리즘의 원리
- [ ] 신경망과 진화의 결합
- [ ] 다양한 최적화 기법

### 실습 능력
- [ ] 파이썬으로 게임 환경 구현
- [ ] 유전 알고리즘 구현
- [ ] 신경망 설계 및 훈련
- [ ] 결과 분석 및 시각화

### 응용 역량
- [ ] 새로운 문제에 알고리즘 적용
- [ ] 하이퍼파라미터 튜닝
- [ ] 성능 개선 방법 탐구
- [ ] 프로젝트 설계 및 구현

## 학습 팁

### 1. 능동적 학습
- 코드를 수정하며 실험해보세요
- 파라미터를 바꿔가며 결과 관찰
- 자신만의 개선 방법 시도

### 2. 시각화 활용
- 그래프와 차트로 결과 분석
- 학습 과정을 눈으로 확인
- 패턴과 트렌드 발견

### 3. 문제 해결
- 오류가 발생하면 차근차근 디버깅
- 다른 설정으로 재시도
- 커뮤니티나 문서 참조

## 문제 해결

### 자주 발생하는 문제들

**Q: 노트북이 실행되지 않아요**
A: 라이브러리 설치 확인 및 경로 설정 점검

**Q: 훈련이 너무 오래 걸려요**
A: 개체군 크기나 세대 수를 줄여보세요

**Q: AI 성능이 향상되지 않아요**
A: 하이퍼파라미터를 조정하거나 더 오래 훈련해보세요

## 추가 자료

### 관련 논문
- Genetic Algorithms in Search, Optimization and Machine Learning
- Evolving Neural Networks through Augmenting Topologies
- Deep Reinforcement Learning: An Overview

### 온라인 자료
- [유전 알고리즘 위키](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [강화학습 개론](https://www.sutton-barto.com/)
- [신경망 기초](https://www.deeplearningbook.org/)

---

**즐거운 학습 되세요!**
