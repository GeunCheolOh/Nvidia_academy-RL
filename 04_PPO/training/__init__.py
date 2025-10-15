"""Training utilities (MultiDiscrete)"""

from .config import get_device, HYPERPARAMS
from .self_play import SelfPlayTrainerMultiDiscrete

__all__ = ['get_device', 'HYPERPARAMS', 'SelfPlayTrainerMultiDiscrete']

