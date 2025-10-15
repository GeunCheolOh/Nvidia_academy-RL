# Pikachu Volleyball ONNX 배포

학습된 PyTorch 모델을 ONNX로 변환하여 다양한 플랫폼에서 사용할 수 있습니다.

## 파일 구조

```
deployment/
├── pikachu_agent.onnx          # 변환된 ONNX 모델 (284KB)
├── export_onnx.py              # PyTorch → ONNX 변환 스크립트
├── test_onnx_inference.py      # ONNX 모델 추론 테스트
└── README.md                   # 이 파일
```

## ONNX 모델 정보

### 입력
- **observation**: `[batch_size, 15]` (float32)
  - 정규화된 게임 상태 (플레이어 위치, 볼 위치 등)

### 출력
- **x_logits**: `[batch_size, 3]` (float32) - x 방향 로짓 (left, stay, right)
- **y_logits**: `[batch_size, 3]` (float32) - y 방향 로짓 (stay, jump, down)
- **power_logits**: `[batch_size, 2]` (float32) - 파워 히트 로짓 (no, yes)
- **value**: `[batch_size, 1]` (float32) - 상태 가치 (사용 안 함)

### 성능
- **파일 크기**: 284KB
- **추론 속도**: PyTorch 대비 2.29배 빠름
- **정확도**: PyTorch와 최대 오차 0.000006 (검증 완료)

## 사용 방법

### 1. PyTorch 모델을 ONNX로 변환

```bash
cd /home/ubuntu/project/alphachu
source venv/bin/activate

# Best model 변환
python deployment/export_onnx.py \
  --model models/checkpoints/best_model.pth \
  --output deployment/pikachu_agent.onnx

# Final model 변환
python deployment/export_onnx.py \
  --model models/checkpoints/final_model.pth \
  --output deployment/pikachu_agent_final.onnx
```

### 2. ONNX 모델 테스트

```bash
# 10 에피소드 테스트 (결정적 정책)
python deployment/test_onnx_inference.py --episodes 10

# 확률적 정책 테스트
python deployment/test_onnx_inference.py --episodes 20 --stochastic
```

## Python에서 ONNX 모델 사용

```python
import numpy as np
import onnxruntime as ort

# 세션 생성
session = ort.InferenceSession("deployment/pikachu_agent.onnx")

# 관찰 (예시)
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

action = np.array([x_action, y_action, power_action])
```

## JavaScript/Web에서 사용 (ONNX.js)

```javascript
// ONNX.js 설치
// npm install onnxjs

const onnx = require('onnxjs');

// 모델 로드
const session = new onnx.InferenceSession();
await session.loadModel('pikachu_agent.onnx');

// 추론
const observation = new Float32Array(15);  // 관찰 데이터
const inputTensor = new onnx.Tensor(observation, 'float32', [1, 15]);

const outputMap = await session.run([inputTensor]);
const xLogits = outputMap.get('x_logits');
const yLogits = outputMap.get('y_logits');
const powerLogits = outputMap.get('power_logits');

// 행동 선택
const xAction = argmax(xLogits.data);
const yAction = argmax(yLogits.data);
const powerAction = argmax(powerLogits.data);
```

## C++에서 사용 (ONNX Runtime)

```cpp
#include <onnxruntime_cxx_api.h>

// 세션 생성
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pikachu");
Ort::SessionOptions session_options;
Ort::Session session(env, "pikachu_agent.onnx", session_options);

// 입력 준비
std::vector<float> observation(15);  // 관찰 데이터
std::vector<int64_t> input_shape = {1, 15};

auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, observation.data(), observation.size(), 
    input_shape.data(), input_shape.size()
);

// 추론
const char* input_names[] = {"observation"};
const char* output_names[] = {"x_logits", "y_logits", "power_logits", "value"};

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr}, 
    input_names, &input_tensor, 1,
    output_names, 4
);

// 결과 사용
float* x_logits = output_tensors[0].GetTensorMutableData<float>();
float* y_logits = output_tensors[1].GetTensorMutableData<float>();
float* power_logits = output_tensors[2].GetTensorMutableData<float>();
```

## 배포 시나리오

### 1. 웹 브라우저
- ONNX.js 또는 ONNX Runtime Web 사용
- 브라우저에서 직접 AI 플레이어 실행

### 2. 모바일 앱
- ONNX Runtime Mobile (iOS/Android)
- 오프라인 AI 플레이어

### 3. 게임 엔진
- Unity ML-Agents (ONNX 지원)
- Unreal Engine (ONNX Runtime 플러그인)

### 4. 임베디드 시스템
- ONNX Runtime (ARM, x86)
- 경량 추론 (284KB)

## 검증 결과

```
PyTorch vs ONNX 출력 비교:
  x_logits: 최대 차이 = 0.00000572
  y_logits: 최대 차이 = 0.00000381
  power_logits: 최대 차이 = 0.00000191
  value: 최대 차이 = 0.00000095

추론 속도 테스트 (100회):
  PyTorch: 12.76 ms
  ONNX: 5.58 ms
  속도 향상: 2.29x

게임 플레이 테스트 (10 에피소드):
  Player 1 평균 점수: 15.00 ± 0.00
  Player 2 평균 점수: 8.30 ± 0.46
  Player 1 승률: 100.0%
```

## 참고 자료

- [ONNX 공식 문서](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [ONNX.js](https://github.com/microsoft/onnxjs)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)

