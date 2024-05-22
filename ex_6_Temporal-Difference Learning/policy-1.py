import numpy as np
from collections import defaultdict
from typing import Callable, Tuple

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])
    # print(num_actions)

    def get_action(state: Tuple) -> int:
        # TODO
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.choice(num_actions)
            # print(action)
        else:
            max_value = np.max(Q[state])
            indices = np.where(Q[state] == max_value)[0]
            action = np.random.choice(indices)
        # print(action)
        return action

    return get_action