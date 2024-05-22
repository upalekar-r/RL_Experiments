# Let's redefine the necessary parts for Monte Carlo Prediction

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from enum import Enum
from tqdm import tqdm
from typing import List, Tuple, Dict, Callable

# Enum for actions in the environment
class Action(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

# Transition function for the environment
def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, 1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, -1),
    }
    return mapping[action]

# Environment simulation function
def simulate(state: Tuple[int, int], action: Action, walls: List[Tuple[int, int]], goal_state: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
    dxdy = actions_to_dxdy(action)
    next_state = (state[0] + dxdy[0], state[1] + dxdy[1])
    if next_state in walls or not (0 <= next_state[0] < 11 and 0 <= next_state[1] < 11):
        next_state = state
    reward = 1.0 if next_state == goal_state else -0.1
    return next_state, reward

# Reset function for the environment
def reset() -> Tuple[int, int]:
    return (0, 0)  # Starting state

# Greedy policy function
def epsilon_greedy_policy(state, Q_values, epsilon=0.1):
    if np.random.rand() < epsilon:
        return Action(np.random.choice(list(Action)))
    else:
        return Action(np.argmax(Q_values[state]))


def plot_V(V, title="Value Function"):
    grid_size = (11, 11)
    V_grid = np.zeros(grid_size)
    for state, value in V.items():
        V_grid[state] = value
    
    plt.figure(figsize=(8, 8))
    im = plt.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Generate episodes following a policy
def generate_episodes(policy, Q_values, walls, goal_state, episodes, max_steps=1000):
    data = []
    for _ in tqdm(range(episodes), desc="Generating episodes"):
        state = reset()
        episode = []
        for _ in range(max_steps):  # Add a step limit
            action = policy(state, Q_values, epsilon=0.1)  # Ensure epsilon is used for some exploration
            next_state, reward = simulate(state, action, walls, goal_state)
            episode.append((state, action, reward))
            state = next_state
            if state == goal_state:
                break
        data.append(episode)
    return data


# On-policy first-visit Monte Carlo prediction algorithm
def on_policy_mc_prediction(episodes_data: List[List[Tuple[Tuple[int, int], Action, float]]], gamma: float) -> Dict[Tuple[int, int], np.ndarray]:
    Q = defaultdict(lambda: np.zeros(len(Action)))
    returns = defaultdict(list)
    
    for episode in episodes_data:
        G = 0
        visited_state_actions = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                returns[(state, action)].append(G)
                # Convert action to its integer value for indexing
                Q[state][action.value] = np.mean(returns[(state, action)])
    return Q


# Define walls and goal state
walls = [
    (0, 5), (2, 5), (3, 5), (4, 5), (5, 0), (5, 2),
    (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9),
    (5, 10), (6, 4), (7, 4), (9, 4), (10, 4),
]
goal_state = (10, 10)

# Run on-policy MC prediction
gamma = 0.9  # Discount factor
episodes = 10000  # Number of episodes to simulate

# Initial Q-values for our policy (randomly initialized)
Q_values = defaultdict(lambda: np.random.random(len(Action)))

# Generate episodes
episodes_data = generate_episodes(epsilon_greedy_policy, Q_values, walls, goal_state, episodes)
# Run on-policy MC prediction
Q_values = on_policy_mc_prediction(episodes_data, gamma)

# Compute V from Q
V = {state: max(action_values) for state, action_values in Q_values.items()}

# Plotting V
plot_V(V, "V from On-policy Monte-Carlo Prediction")
