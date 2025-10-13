# CLAUDE.md - 프로젝트 규칙 및 명령어 가이드

## 안전 규칙
- Plan Mode에서 계획 먼저 제시하고 승인을 받기 전에는 파일 편집/명령 실행 금지
- 필요 시 `/permissions`로 안전한 툴만 허용
- YOLO(무허가 실행) 모드는 절대 사용 금지

## 프로젝트 구조
```
atari/
├── notebooks/
│   ├── 01_Qlearning_Theory_Workflow.ipynb
│   └── 02_Qlearning_FrozenLake_Training.ipynb
├── scripts/
│   ├── frozenlake_keyboard_agent.py
│   ├── q_learning_train.py
│   └── q_learning_eval.py
├── weights/
│   └── q_table.npy
├── utils/
│   ├── plotting.py
│   └── io.py
├── venv/
├── CLAUDE.md
├── README.md
└── requirements.txt
```

## 환경 설정 명령어
```bash
# Conda 환경 (권장)
conda create -n frozenlake python=3.10 -y
conda activate frozenlake
pip install -r requirements.txt

# 또는 Python venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# 가상환경 비활성화
conda deactivate  # conda 사용 시
deactivate        # venv 사용 시
```

## 실행 명령어
```bash
# 키보드 조작 테스트 (슬리피 모드 OFF)
python scripts/frozenlake_keyboard_agent.py

# Q-Learning 훈련
python scripts/q_learning_train.py --episodes 5000  # 4x4 고정 맵 -> q_table_4x4.npy
python scripts/q_learning_train.py --episodes 10000 --map 8x8  # 8x8 맵 -> q_table_8x8.npy
python scripts/q_learning_train.py --episodes 5000 --random-map  # 랜덤 맵 -> q_table_4x4_random.npy

# 학습된 모델 평가
python scripts/q_learning_eval.py --episodes 100  # 기본: q_table_4x4.npy
python scripts/q_learning_eval.py --demonstrate  # 시각적 시연
```

## 작업 절차
1. Explore: 코드베이스 탐색 및 이해
2. Plan: 변경사항 계획 수립 및 승인
3. Code: 구현 및 테스트
4. Verify: 검증 및 문서화

## 공통 규칙
- matplotlib만 사용, seaborn 금지
- 한 차트 = 한 Figure
- 색상 수동 지정 금지
- 난수 시드 고정 (np.random.seed(42))
- 재현 가능한 산출물 보장