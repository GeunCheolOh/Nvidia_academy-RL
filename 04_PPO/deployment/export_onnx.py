"""
학습된 PyTorch 모델을 ONNX로 변환 (MultiDiscrete Action Space)
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from agents.network import ActorCriticNetworkMultiDiscrete
from agents.ppo import PPOAgentMultiDiscrete


def export_to_onnx(
    model_path: str = "models/checkpoints/best_model.pth",
    output_path: str = "deployment/pikachu_agent.onnx",
    observation_dim: int = 15,
):
    """
    PyTorch 모델을 ONNX로 변환
    
    Args:
        model_path: 학습된 모델 경로
        output_path: ONNX 파일 저장 경로
        observation_dim: 관찰 차원
    """
    print("=" * 60)
    print("PyTorch to ONNX 변환")
    print("=" * 60)
    
    # 모델 로드
    print(f"\n모델 로드: {model_path}")
    agent = PPOAgentMultiDiscrete(observation_dim=observation_dim, device="cpu")
    agent.load(model_path)
    agent.network.eval()
    
    # 더미 입력 (배치 크기 1)
    dummy_input = torch.randn(1, observation_dim)
    
    print(f"\n입력 shape: {dummy_input.shape}")
    print(f"출력: x_logits(3), y_logits(3), power_logits(2), value(1)")
    
    # ONNX 변환
    print(f"\nONNX 변환 중...")
    torch.onnx.export(
        agent.network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['x_logits', 'y_logits', 'power_logits', 'value'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'x_logits': {0: 'batch_size'},
            'y_logits': {0: 'batch_size'},
            'power_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX 모델 저장 완료: {output_path}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(output_path) / 1024
    print(f"파일 크기: {file_size:.2f} KB")
    
    # 검증
    print("\nONNX 모델 검증 중...")
    try:
        import onnxruntime as ort
        
        ort_session = ort.InferenceSession(output_path)
        
        # PyTorch 출력
        with torch.no_grad():
            x_logits_pt, y_logits_pt, power_logits_pt, value_pt = agent.network(dummy_input)
            pytorch_outputs = [
                x_logits_pt.numpy(),
                y_logits_pt.numpy(),
                power_logits_pt.numpy(),
                value_pt.numpy()
            ]
        
        # ONNX 출력
        onnx_outputs = ort_session.run(None, {'observation': dummy_input.numpy()})
        
        # 비교
        print("\nPyTorch vs ONNX 출력 비교:")
        output_names = ['x_logits', 'y_logits', 'power_logits', 'value']
        max_diff = 0.0
        
        for i, name in enumerate(output_names):
            diff = abs(pytorch_outputs[i] - onnx_outputs[i]).max()
            max_diff = max(max_diff, diff)
            print(f"  {name}: 최대 차이 = {diff:.8f}")
        
        print(f"\n전체 최대 차이: {max_diff:.8f}")
        
        if max_diff < 1e-5:
            print("ONNX 변환 검증 성공!")
        else:
            print("경고: ONNX 변환 검증 - 출력 차이가 있습니다")
        
        # 추론 속도 테스트
        print("\n추론 속도 테스트 (100회)...")
        import time
        
        # PyTorch
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                agent.network(dummy_input)
        pytorch_time = (time.time() - start) * 1000
        
        # ONNX
        start = time.time()
        for _ in range(100):
            ort_session.run(None, {'observation': dummy_input.numpy()})
        onnx_time = (time.time() - start) * 1000
        
        print(f"  PyTorch: {pytorch_time:.2f} ms (100회)")
        print(f"  ONNX: {onnx_time:.2f} ms (100회)")
        print(f"  속도 향상: {pytorch_time/onnx_time:.2f}x")
        
    except ImportError:
        print("경고: onnxruntime이 설치되지 않아 검증을 건너뜁니다")
        print("설치: pip install onnxruntime")
    
    print("\n" + "=" * 60)
    print("변환 완료!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch 모델을 ONNX로 변환")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_model.pth", help="PyTorch 모델 경로")
    parser.add_argument("--output", type=str, default="deployment/pikachu_agent.onnx", help="ONNX 출력 경로")
    parser.add_argument("--observation-dim", type=int, default=15, help="관찰 차원")
    args = parser.parse_args()
    
    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        observation_dim=args.observation_dim
    )

