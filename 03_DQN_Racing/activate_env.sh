#!/bin/bash
echo "Activating DQN Racing environment..."
source /Users/ogeuncheol/Documents/project/RL/2_1_DQN_Racing/dqn_racing_env/bin/activate
echo "Environment activated! You can now run the training scripts."
echo ""
echo "Available commands:"
echo "  python games/test_manual_play.py       - Test manual gameplay"
echo "  python tutorials/dqn_tutorial.py       - Run DQN tutorial"
echo "  python training/dqn_training.py        - Start training"
echo "  python games/demo_trained_agent.py     - Demo trained agent"
exec "$SHELL"
