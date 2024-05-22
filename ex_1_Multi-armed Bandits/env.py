import numpy as np
from typing import Tuple


class BanditEnv:
    """Multi-armed bandit environment"""

    def __init__(self, k: int) -> None:
        """__init__.

        Args:
            k (int): number of arms/bandits
        """
        self.k = k+1
        self.optimal_action = None

    def reset(self) -> None:
        """Resets the mean payout/reward of each arm.
        This function should be called at least once after __init__()
        """
        # Initialize means of each arm distributed according to standard normal
        self.means = np.random.normal(size=self.k)
        # Update optimal action
        self.optimal_action = np.argmax(self.means)
        # maximum reward for action
        self.max_action = np.max(self.means)

    def step(self, action: int) -> Tuple[float, int]:
        """Take one step in env (pull one arm) and observe reward

        Args:
            action (int): index of arm to pull
        """
        # TODO calculate reward of arm given by action
        reward = np.random.normal(self.means[action])

        return reward
    
