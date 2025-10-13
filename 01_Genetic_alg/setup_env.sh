#!/bin/bash

echo "========================================="
echo "Snake Genetic Algorithm - 환경 설치"
echo "========================================="
echo ""

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "작업 디렉토리: $SCRIPT_DIR"
echo ""

# Conda 설치 확인
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda를 찾을 수 없습니다."
    echo ""
    echo "Miniconda 또는 Anaconda를 먼저 설치해주세요:"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "  - Anaconda: https://www.anaconda.com/download"
    echo ""
    exit 1
fi

echo "[OK] conda 발견: $(conda --version)"
echo ""

# 환경 이름 설정
ENV_NAME="snake_ga"

# 기존 환경 확인
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[WARNING] 기존 환경 '${ENV_NAME}'이(가) 발견되었습니다."
    read -p "삭제하고 새로 만드시겠습니까? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "기존 환경 삭제 중..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "설치를 취소했습니다."
        exit 1
    fi
fi

# Conda 환경 생성
echo "Conda 환경 생성 중 (Python 3.9)..."
conda create -n ${ENV_NAME} python=3.9 -y

if [ $? -ne 0 ]; then
    echo "[ERROR] 환경 생성에 실패했습니다."
    exit 1
fi

echo "[OK] 환경 생성 완료"
echo ""

# 환경 활성화
echo "환경 활성화 중..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

if [ $? -ne 0 ]; then
    echo "[ERROR] 환경 활성화에 실패했습니다."
    exit 1
fi

echo "[OK] 환경 활성화 완료"
echo ""

# 패키지 설치
echo "라이브러리 설치 중..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] 라이브러리 설치에 실패했습니다."
        exit 1
    fi
else
    echo "[WARNING] requirements.txt 파일을 찾을 수 없습니다."
fi

# 설치 완료
echo ""
echo "========================================="
echo "[OK] 설치가 완료되었습니다!"
echo "========================================="
echo ""
echo "환경을 활성화하려면:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "환경을 비활성화하려면:"
echo "  conda deactivate"
echo ""
echo "훈련 시작:"
echo "  python train_ga.py --population 30 --generations 50"
echo ""
echo "모델 테스트:"
echo "  python test_agent.py --model models/snake_ga_best.npz"
echo ""
