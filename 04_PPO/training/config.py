"""설정 및 하이퍼파라미터"""
import torch


def get_device():
    """디바이스 자동 선택: CUDA → MPS → CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# PPO 하이퍼파라미터
HYPERPARAMS = {
    # 환경
    "winning_score": 15,
    
    # 네트워크
    "observation_dim": 15,
    "action_dim": 9,
    "hidden_dims": (256, 256),
    
    # PPO
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "normalize_advantages": True,
    
    # 학습
    "n_steps": 2048,  # Rollout 길이
    "n_epochs": 10,   # PPO 에포크
    "batch_size": 64,
    
    # 스케줄
    "total_timesteps": 1_000_000,  # 총 학습 스텝
    "save_freq": 10_000,           # 저장 주기
    "eval_freq": 10_000,           # 평가 주기
    "log_freq": 1_000,             # 로그 주기
}

