"""
PPO Self-Play 학습 스크립트 (Multi-Discrete Action Space)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from env.pikachu_env import PikachuVolleyballEnvMultiDiscrete
from agents.ppo import PPOAgentMultiDiscrete
from training.self_play import SelfPlayTrainerMultiDiscrete
from training import get_device, HYPERPARAMS


def main():
    parser = argparse.ArgumentParser(description="Pikachu Volleyball PPO Training (MultiDiscrete)")
    parser.add_argument("--resume", type=str, default=None, help="체크포인트 경로 (재개)")
    parser.add_argument("--timesteps", type=int, default=HYPERPARAMS["total_timesteps"], help="총 학습 스텝")
    parser.add_argument("--save-dir", type=str, default="models/checkpoints", help="저장 디렉토리")
    parser.add_argument("--log-dir", type=str, default="logs/tensorboard", help="로그 디렉토리")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pikachu Volleyball PPO Self-Play 학습 (MultiDiscrete)")
    print("=" * 60)
    
    # 디바이스 설정
    device = get_device()
    
    # 환경 생성
    env = PikachuVolleyballEnvMultiDiscrete(winning_score=HYPERPARAMS["winning_score"])
    print(f"\n환경:")
    print(f"  관찰 공간: {env.observation_space}")
    print(f"  행동 공간: {env.action_space} (3*3*2 = 18 조합)")
    
    # 에이전트 생성
    agent = PPOAgentMultiDiscrete(
        observation_dim=HYPERPARAMS["observation_dim"],
        device=str(device),
        learning_rate=HYPERPARAMS["learning_rate"],
        gamma=HYPERPARAMS["gamma"],
        gae_lambda=HYPERPARAMS["gae_lambda"],
        clip_epsilon=HYPERPARAMS["clip_epsilon"],
        value_coef=HYPERPARAMS["value_coef"],
        entropy_coef=HYPERPARAMS["entropy_coef"],
        max_grad_norm=HYPERPARAMS["max_grad_norm"],
        normalize_advantages=HYPERPARAMS["normalize_advantages"],
    )
    
    print(f"\nPPO 에이전트:")
    print(f"  파라미터: {sum(p.numel() for p in agent.network.parameters()):,}")
    print(f"  Learning rate: {HYPERPARAMS['learning_rate']}")
    print(f"  Entropy coef: {HYPERPARAMS['entropy_coef']}")
    
    # 트레이너 생성
    trainer = SelfPlayTrainerMultiDiscrete(
        env=env,
        agent=agent,
        n_steps=HYPERPARAMS["n_steps"],
        gamma=HYPERPARAMS["gamma"],
        gae_lambda=HYPERPARAMS["gae_lambda"],
        device=str(device),
    )
    
    print(f"\n학습 설정:")
    print(f"  Total timesteps: {args.timesteps:,}")
    print(f"  Rollout steps: {HYPERPARAMS['n_steps']}")
    print(f"  PPO epochs: {HYPERPARAMS['n_epochs']}")
    
    if args.resume:
        print(f"\n체크포인트에서 재개: {args.resume}")
    
    print("\n학습 시작...")
    print("=" * 60)
    
    # 학습
    trainer.train(
        total_timesteps=args.timesteps,
        n_epochs=HYPERPARAMS["n_epochs"],
        batch_size=HYPERPARAMS["batch_size"],
        save_freq=HYPERPARAMS["save_freq"],
        eval_freq=HYPERPARAMS["eval_freq"],
        log_freq=HYPERPARAMS["log_freq"],
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_from=args.resume,
    )
    
    print("\n=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"\n모델 저장 위치: {args.save_dir}/final_model.pth")
    print(f"텐서보드 로그: {args.log_dir}")
    print("\n텐서보드 실행:")
    print(f"  tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()

