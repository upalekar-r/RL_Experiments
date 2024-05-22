import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from typing import Tuple, List, Dict, Callable
from tqdm import tqdm  # Progress bar for loops
from collections import defaultdict

# Enum for actions in the environment
class Action(IntEnum):
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
    reward = 1.0 if next_state == goal_state else 0.0
    return next_state, reward

# Reset function for the environment
def reset() -> Tuple[int, int]:
    return (0, 0)  # Starting state

# Random policy function
def random_policy(state: Tuple[int, int]) -> Action:
    return Action(np.random.choice(list(Action)))

# Generate episodes following a policy
def generate_episodes(policy: Callable[[Tuple[int, int]], Action], walls: List[Tuple[int, int]], goal_state: Tuple[int, int], episodes: int) -> List[List[Tuple[Tuple[int, int], Action, float]]]:
    data = []
    for _ in tqdm(range(episodes), desc="Generating episodes"):
        state = reset()
        episode = []
        while state != goal_state:
            action = policy(state)
            next_state, reward = simulate(state, action, walls, goal_state)
            episode.append((state, action, reward))
            state = next_state
        data.append(episode)
    return data

# Monte Carlo prediction function for off-policy evaluation
def off_policy_mc_prediction(episodes_data: List[List[Tuple[Tuple[int, int], Action, float]]], gamma: float) -> Dict[Tuple[int, int], float]:
    Q = defaultdict(lambda: np.zeros(4))  # Q-function
    C = defaultdict(lambda: np.zeros(4))  # Cumulative weights
    # Assume b(A|S) is uniform and π(A|S) is greedy with respect to Q
    for episode in episodes_data:
        G = 0
        W = 1
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action != np.argmax(Q[state]):  # If not taken by target policy
                break
            W = W * 1 / 0.25  # 4 actions => b(A|S) = 0.25
    return Q

# Define the walls and goal state of the Four Rooms environment
walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
    ] 
    # Add the specific walls of the Four Rooms environment

goal_state = (10, 10)

# Main execution
if __name__ == "__main__":
    episodes_random = generate_episodes(random_policy, walls, goal_state, 10000)
    # Estimate Q-values using the generated episodes
    Q_values = off_policy_mc_prediction(episodes_random, gamma=1.0)
    
    # Print the estimated Q-values
    for state, values in Q_values.items():
        print(f"State: {state}, Q-values: {values}")
    # Further steps would include generating episodes from the Monte Carlo control policy and the greedy policy,
    # then using off_policy_mc_prediction to estimate Q-values.
