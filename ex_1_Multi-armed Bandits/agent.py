import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Sequence
import random
import math

def argmax(arr: Sequence[float]) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    # TODO
    # if not arr:
    #     raise ValueError("Input array is empty")

    max_value = max(arr)
    indices_with_max_value = [i for i, value in enumerate(arr) if value == max_value]
    selected_index = random.choice(indices_with_max_value)

    return selected_index
    pass


class BanditAgent(ABC):
    def __init__(self, k: int, init: int, step_size: float) -> None:
        """Abstract bandit agent class

        Implements common functions for both epsilon greedy and UCB

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            step_size (float): step size
        """
        self.k = k
        self.init = init
        self.step_size = step_size

        # Q-values for each arm
        self.Q = [init] * k
        # Number of times each arm was pulled
        self.N = [0] * k
        # Current total number of steps
        self.t = 0

    def reset(self) -> None:
        """Initialize or reset Q-values and counts

        This method should be called after __init__() at least once
        """
        self.Q = self.init * np.ones(self.k, dtype=np.float32)
        self.N = np.zeros(self.k, dtype=int)
        self.t = 0

    @abstractmethod
    def choose_action(self) -> int:
        """Choose which arm to pull"""
        raise NotImplementedError

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        raise NotImplementedError

class EpsilonGreedy(BanditAgent):
    def __init__(
        self, k: int, init: int, epsilon: float, step_size: Optional[float] = None
    ) -> None:
        """Epsilon greedy bandit agent

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            epsilon (float): random action probability
            step_size (float or None): step size. If None, then it is equal to 1 / N_t (dynamic step size)
        """
        super().__init__(k, init, step_size)
        self.epsilon = epsilon
    
    def choose_action(self):
        """Choose which arm to pull

        With probability 1 - epsilon, choose the best action (break ties arbitrarily, use argmax() from above). With probability epsilon, choose a random action.
        """
        # TODO
        # p = random.randint(0, self.k - 1)
        # q = argmax(self.Q)
        # action_list = [p,q]
        # action = random.choices(action_list , weights = (self.epsilon, (1-self.epsilon)))[0] # another method
        if random.random() < self.epsilon:
            # Choose a random action
            action = random.randint(0, self.k - 1)
        else:
            # Choose the best action based on Q-values (break ties randomly)
            action = argmax(self.Q)

        # # Increment the total step count
        # self.t += 1
        return action
    
    def optimal_action_count(self, action)-> int:
        return 1 if action == argmax(self.Q) else 0


    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        self.t += 1

        # TODO update self.N
        self.N[action] += 1
        # TODO update self.Q
        # If step_size is given (static step size)
        if self.step_size is not None:
            self.Q[action] += self.step_size * (reward - self.Q[action])
            #pass
        # If step_size is dynamic (step_size = 1 / N(a))
        else:
             self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])
        pass


class UCB(BanditAgent):
    def __init__(self, k: int, init: int, c: float, step_size: Optional[float]) -> None:
        """Epsilon greedy bandit agent

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            c (float): UCB constant that controls degree of exploration
            step_size (float): step size (use constant step size in case of UCB)
        """
        super().__init__(k, init, step_size)
        self.c = c

    def choose_action(self):
        """Choose which arm to pull

        Use UCB action selection. Be sure to consider the case when N_t = 0 and break ties randomly (use argmax() from above)
        """
        # TODO
        if 0 in self.N:
            # Choose an unexplored arm (if any)
            unexplored_arms = [i for i, n in enumerate(self.N) if n == 0]
            action = random.choice(unexplored_arms)
        else:
            # Choose action using UCB formula
            ucb_values = [self.Q[a] + self.c * math.sqrt(math.log(self.t) / self.N[a]) for a in range(self.k)]
            action = argmax(ucb_values)

        return action

    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        self.t += 1

        # TODO update self.N
        self.N[action] += 1

        # TODO update self.Q
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])