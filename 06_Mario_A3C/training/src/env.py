"""
Environment wrapper for Super Mario Bros training.
This module provides the environment setup with preprocessing wrappers.
"""

import gym_super_mario_bros
from gym import Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np


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
    Custom reward shaping for better learning signal.

    Rewards:
    - Progress reward: Based on score increase
    - Flag reward: +50 for completing the level
    - Death penalty: -50 for dying
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
        done = False

        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)

        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)


def create_train_env(world, stage, action_type):
    """
    Create the Super Mario Bros training environment with all preprocessing wrappers.

    Args:
        world: World number (1-8)
        stage: Stage number (1-4)
        action_type: Action space type:
            - "right": Only rightward movements
            - "simple": Simplified action set
            - "complex": Full action set

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
    env = CustomSkipFrame(env)

    return env, env.observation_space.shape[0], len(actions)
