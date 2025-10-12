#!/bin/bash

echo "========================================="
echo "1_1_Genetic_alg 가상환경 설치 스크립트"
echo "========================================="
echo ""

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "작업 디렉토리: $SCRIPT_DIR"
echo ""

# 기존 가상환경 확인
if [ -d "venv" ]; then
    echo "[WARNING] 기존 가상환경이 발견되었습니다."
    read -p "삭제하고 새로 만드시겠습니까? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "기존 가상환경 삭제 중..."
        rm -rf venv
    else
        echo "[ERROR] 설치를 취소했습니다."
        exit 1
    fi
fi

# Python 버전 확인
echo "Python 버전 확인 중..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version)
    echo "   $PYTHON_VERSION 발견"
else
    echo "[ERROR] python3를 찾을 수 없습니다."
    echo "   Python 3.7 이상을 설치해주세요."
    exit 1
fi

# 가상환경 생성
echo ""
echo "가상환경 생성 중..."
$PYTHON_CMD -m venv venv

if [ $? -ne 0 ]; then
    echo "[ERROR] 가상환경 생성에 실패했습니다."
    exit 1
fi

echo "[OK] 가상환경 생성 완료"

# 가상환경 활성화
echo ""
echo "가상환경 활성화 중..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "[ERROR] 가상환경 활성화에 실패했습니다."
    exit 1
fi

echo "[OK] 가상환경 활성화 완료"

# pip 업그레이드
echo ""
echo "pip 업그레이드 중..."
pip install --upgrade pip

# requirements.txt 설치
echo ""
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
echo "가상환경을 활성화하려면:"
echo "  source venv/bin/activate"
echo ""
echo "가상환경을 비활성화하려면:"
echo "  deactivate"
echo ""
echo "훈련 시작:"
echo "  python train.py --generations 50 --population_size 30 --visualize"
echo ""
echo "모델 플레이:"
echo "  python play.py --model models/snake_ga_best.npz --episodes 5"
echo ""

