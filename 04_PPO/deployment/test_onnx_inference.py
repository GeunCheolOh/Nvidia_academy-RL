"""
ONNX 모델 추론 테스트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort
from env.pikachu_env import PikachuVolleyballEnvMultiDiscrete
from training.self_play import mirror_action_multidiscrete


def softmax(x):
    """Softmax 함수"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def select_action_from_logits(x_logits, y_logits, power_logits, deterministic=True):
    """로짓에서 행동 선택"""
    if deterministic:
        # Greedy
        x = np.argmax(x_logits)
        y = np.argmax(y_logits)
        power = np.argmax(power_logits)
    else:
        # 확률적 샘플링
        x_probs = softmax(x_logits)
        y_probs = softmax(y_logits)
        power_probs = softmax(power_logits)
        
        x = np.random.choice(3, p=x_probs[0])
        y = np.random.choice(3, p=y_probs[0])
        power = np.random.choice(2, p=power_probs[0])
    
    return np.array([x, y, power])


def test_onnx_inference(
    onnx_path: str = "deployment/pikachu_agent.onnx",
    episodes: int = 5,
    deterministic: bool = True
):
    """ONNX 모델로 게임 플레이 테스트"""
    print("=" * 60)
    print("ONNX 모델 추론 테스트")
    print("=" * 60)
    
    # ONNX 세션 생성
    print(f"\nONNX 모델 로드: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
    
    # 입력/출력 정보
    print("\n입력:")
    for input_meta in ort_session.get_inputs():
        print(f"  {input_meta.name}: {input_meta.shape} ({input_meta.type})")
    
    print("\n출력:")
    for output_meta in ort_session.get_outputs():
        print(f"  {output_meta.name}: {output_meta.shape} ({output_meta.type})")
    
    # 환경 생성
    env = PikachuVolleyballEnvMultiDiscrete(winning_score=15)
    
    print(f"\n게임 설정:")
    print(f"  에피소드: {episodes}")
    print(f"  정책: {'결정적' if deterministic else '확률적'}")
    
    # 통계
    scores_p1 = []
    scores_p2 = []
    wins_p1 = 0
    
    for ep in range(episodes):
        (obs_p1, obs_p2), _ = env.reset()
        done = False
        
        while not done:
            # Player 1 행동 선택
            obs_p1_input = obs_p1.reshape(1, -1).astype(np.float32)
            x_logits, y_logits, power_logits, value = ort_session.run(
                None, {'observation': obs_p1_input}
            )
            action_p1 = select_action_from_logits(x_logits, y_logits, power_logits, deterministic)
            
            # Player 2 행동 선택
            obs_p2_input = obs_p2.reshape(1, -1).astype(np.float32)
            x_logits, y_logits, power_logits, value = ort_session.run(
                None, {'observation': obs_p2_input}
            )
            action_p2_mirrored = select_action_from_logits(x_logits, y_logits, power_logits, deterministic)
            action_p2 = mirror_action_multidiscrete(action_p2_mirrored)
            
            # 환경 진행
            (obs_p1, obs_p2), _, terminated, truncated, info = env.step((action_p1, action_p2))
            done = terminated or truncated
        
        scores_p1.append(info['score_p1'])
        scores_p2.append(info['score_p2'])
        if info['score_p1'] > info['score_p2']:
            wins_p1 += 1
        
        print(f"  게임 {ep+1}: {info['score_p1']}-{info['score_p2']}")
    
    env.close()
    
    # 결과
    print("\n" + "=" * 60)
    print("테스트 결과")
    print("=" * 60)
    print(f"총 게임: {episodes}")
    print(f"Player 1 평균 점수: {np.mean(scores_p1):.2f} ± {np.std(scores_p1):.2f}")
    print(f"Player 2 평균 점수: {np.mean(scores_p2):.2f} ± {np.std(scores_p2):.2f}")
    print(f"Player 1 승률: {wins_p1}/{episodes} ({wins_p1/episodes*100:.1f}%)")
    print("=" * 60)
    print("\nONNX 모델이 정상적으로 작동합니다!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX 모델 추론 테스트")
    parser.add_argument("--onnx", type=str, default="deployment/pikachu_agent.onnx", help="ONNX 모델 경로")
    parser.add_argument("--episodes", type=int, default=5, help="테스트 에피소드 수")
    parser.add_argument("--stochastic", action="store_true", help="확률적 정책 사용")
    args = parser.parse_args()
    
    test_onnx_inference(
        onnx_path=args.onnx,
        episodes=args.episodes,
        deterministic=not args.stochastic
    )

