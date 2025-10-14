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
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten and pass through LSTM
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))

        # Actor and Critic outputs
        actor_logits = self.actor_linear(hx)
        critic_value = self.critic_linear(hx)

        return actor_logits, critic_value, hx, cx
