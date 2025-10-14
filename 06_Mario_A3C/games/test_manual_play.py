"""
Manual play script for Super Mario Bros with keyboard controls.
This script allows you to play the game using keyboard controls.

Controls:
- Arrow Keys: Move (Left/Right), Duck (Down)
- Space: Jump
- A: Run/Fire
- Q: Quit game
"""

import sys
import argparse
import time
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym import Wrapper
from gym.spaces import Box
import cv2
import numpy as np
import pygame


def process_frame(frame):
    """
    Process a single frame: convert to grayscale and resize.

    Args:
        frame: Raw RGB frame from the environment

    Returns:
        Processed grayscale frame of shape (1, 84, 84)
    """
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.0
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    """
    Custom reward shaping for better learning.
    This wrapper adds progress-based rewards and terminal rewards.
    """

    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)

        # Progress reward based on score increase
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]

        # Terminal rewards
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50

        return state, reward / 10.0, done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    """
    Frame skip wrapper that repeats actions and stacks frames.
    """

    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)

        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)

        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)


def create_env(world, stage, action_type):
    """
    Create the Super Mario Bros environment with wrappers.

    Args:
        world: World number (1-8)
        stage: Stage number (1-4)
        action_type: Action space type (right/simple/complex)

    Returns:
        Tuple of (env, num_states, num_actions)
    """
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")

    # Remove TimeLimit wrapper to avoid step API incompatibility
    if hasattr(env, 'env'):
        env = env.env

    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    env = JoypadSpace(env, actions)
    env = CustomReward(env)
    # Use skip=1 for manual play to make it slower and more controllable
    env = CustomSkipFrame(env, skip=1)

    return env, env.observation_space.shape[0], len(actions)


def get_keyboard_action(action_type):
    """
    Get action from keyboard input using pygame.

    Returns:
        Action index based on keyboard state
    """
    keys = pygame.key.get_pressed()

    # Default action: NOOP (0)
    action = 0

    if action_type == "complex":
        # COMPLEX_MOVEMENT actions (12 actions):
        # 0: NOOP
        # 1: right
        # 2: right + A (run)
        # 3: right + B (jump)
        # 4: right + A + B (run + jump)
        # 5: A
        # 6: left
        # 7: left + A
        # 8: left + B
        # 9: left + A + B
        # 10: down
        # 11: up

        right = keys[pygame.K_RIGHT]
        left = keys[pygame.K_LEFT]
        down = keys[pygame.K_DOWN]
        up = keys[pygame.K_UP]
        a_button = keys[pygame.K_a]  # Run
        b_button = keys[pygame.K_SPACE]  # Jump

        if right and a_button and b_button:
            action = 4  # right + A + B
        elif right and b_button:
            action = 3  # right + B (jump)
        elif right and a_button:
            action = 2  # right + A (run)
        elif right:
            action = 1  # right
        elif left and a_button and b_button:
            action = 9  # left + A + B
        elif left and b_button:
            action = 8  # left + B
        elif left and a_button:
            action = 7  # left + A
        elif left:
            action = 6  # left
        elif a_button:
            action = 5  # A
        elif down:
            action = 10  # down
        elif up:
            action = 11  # up

    elif action_type == "simple":
        # SIMPLE_MOVEMENT actions (7 actions):
        # 0: NOOP
        # 1: right
        # 2: right + A (run/jump)
        # 3: right + B (jump)
        # 4: right + A + B
        # 5: A
        # 6: left

        right = keys[pygame.K_RIGHT]
        left = keys[pygame.K_LEFT]
        a_button = keys[pygame.K_a]
        b_button = keys[pygame.K_SPACE]

        if right and a_button and b_button:
            action = 4
        elif right and b_button:
            action = 3
        elif right and a_button:
            action = 2
        elif right:
            action = 1
        elif left:
            action = 6
        elif a_button:
            action = 5

    else:  # "right"
        # RIGHT_ONLY actions (5 actions):
        # 0: NOOP
        # 1: right
        # 2: right + A
        # 3: right + B
        # 4: right + A + B

        right = keys[pygame.K_RIGHT]
        a_button = keys[pygame.K_a]
        b_button = keys[pygame.K_SPACE]

        if right and a_button and b_button:
            action = 4
        elif right and b_button:
            action = 3
        elif right and a_button:
            action = 2
        elif right:
            action = 1

    return action


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play Super Mario Bros with keyboard")
    parser.add_argument("--world", type=int, default=1, help="World number (1-8)")
    parser.add_argument("--stage", type=int, default=1, help="Stage number (1-4)")
    parser.add_argument("--action_type", type=str, default="complex",
                       choices=["right", "simple", "complex"],
                       help="Action space type")

    args = parser.parse_args()

    if not (1 <= args.world <= 8):
        print("Error: World must be between 1 and 8")
        sys.exit(1)
    if not (1 <= args.stage <= 4):
        print("Error: Stage must be between 1 and 4")
        sys.exit(1)

    # Initialize pygame for keyboard input
    pygame.init()

    print("\n" + "="*60)
    print("SUPER MARIO BROS - MANUAL PLAY")
    print("="*60)
    print(f"\nWorld: {args.world}-{args.stage}")
    print(f"Action type: {args.action_type}")
    print("\nControls:")
    print("  Arrow Keys: Move (Left/Right), Duck (Down)")
    print("  Space:      Jump")
    print("  A:          Run/Fire")
    print("  Q or ESC:   Quit game")
    print("="*60 + "\n")

    # Create environment
    env, num_states, num_actions = create_env(args.world, args.stage, args.action_type)

    print(f"Environment created successfully!")
    print(f"  State shape: {env.observation_space.shape}")
    print(f"  Number of actions: {num_actions}")
    print(f"\nStarting game... Press keys to play!\n")

    # Play the game
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    try:
        while not done:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        done = True

            if done:
                break

            # Render the game
            env.render()

            # Get action from keyboard
            action = get_keyboard_action(args.action_type)

            # Take action
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Add delay to slow down gameplay (30 FPS)
            time.sleep(0.033)

            # Display info every 100 steps
            if step_count % 100 == 0:
                print(f"Step: {step_count:4d} | Score: {info['score']:6d} | "
                      f"Reward: {total_reward:7.2f} | Position: {info['x_pos']:4d}")

            # Check for completion
            if info.get("flag_get", False):
                print(f"\n{'='*60}")
                print(f"🎉 LEVEL COMPLETED! 🎉")
                print(f"{'='*60}")
                break

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")

    finally:
        env.close()
        pygame.quit()

        print(f"\nGame Over!")
        print(f"Final Stats:")
        print(f"  Steps: {step_count}")
        print(f"  Score: {info.get('score', 0)}")
        print(f"  Total Reward: {total_reward:.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
