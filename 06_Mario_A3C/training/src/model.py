"""
Actor-Critic neural network model for A2C.
This model uses a shared CNN backbone with separate Actor and Critic heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared convolutional layers.

    Architecture:
    - 4 convolutional layers for feature extraction
    - LSTM layer for temporal dependencies
    - Separate Actor and Critic heads
    """

    def __init__(self, num_inputs, num_actions):
        """
        Initialize the Actor-Critic network.

        Args:
            num_inputs: Number of input channels (frame stack size)
            num_actions: Number of possible actions
        """
        super(ActorCritic, self).__init__()

        # Shared convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)

        # Actor head: outputs action probabilities
        self.actor_linear = nn.Linear(512, num_actions)

        # Critic head: outputs state value
        self.critic_linear = nn.Linear(512, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        """
        Forward pass through the network.

        Args:
            x: Input state (batch_size, num_inputs, 84, 84)
            hx: Hidden state of LSTM
            cx: Cell state of LSTM

        Returns:
            Tuple of (actor_logits, critic_value, new_hx, new_cx)
        """
        # TODO: Actor-Critic 네트워크의 forward pass를 구현하세요
        # 힌트 1: Conv 레이어들(conv1~4)을 순차적으로 통과시키고 F.relu 적용
        # 힌트 2: x.view로 flatten하여 LSTM 입력 준비
        # 힌트 3: LSTM을 통과시켜 temporal dependency 학습 (hidden/cell state 업데이트)
        # 힌트 4: actor_linear로 행동 확률(logits) 출력
        # 힌트 5: critic_linear로 상태 가치(value) 출력
        #YOUR CODE HERE
        raise NotImplementedError("Actor-Critic forward pass를 구현하세요")
