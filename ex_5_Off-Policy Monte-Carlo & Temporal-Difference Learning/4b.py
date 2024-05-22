import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from typing import Tuple, List, Dict, Callable
from collections import defaultdict
from tqdm import tqdm  # Progress bar for loops

class Action(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    

def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, 1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, -1),
    }
    return mapping[action]

def simulate(state: Tuple[int, int], action: Action, walls: List[Tuple[int, int]], goal_state: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
    dxdy = actions_to_dxdy(action)
    next_state = (state[0] + dxdy[0], state[1] + dxdy[1])
    if next_state in walls or not (0 <= next_state[0] < 11 and 0 <= next_state[1] < 11):
        next_state = state
        reward = -0.1  # Minor penalty for hitting a wall or attempting to leave the grid
    else:
        reward = 1.0 if next_state == goal_state else -0.1  # Small negative reward for each step
    return next_state, reward

def reset() -> Tuple[int, int]:
    return (0, 0)

def random_policy(state: Tuple[int, int]) -> Action:
    return Action(np.random.choice(list(Action)))

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

def off_policy_mc_prediction(episodes_data: List[List[Tuple[Tuple[int, int], Action, float]]], gamma: float) -> Dict[Tuple[int, int], float]:
    Q = defaultdict(lambda: np.zeros(4))
    C = defaultdict(lambda: np.zeros(4))
    for episode in episodes_data:
        G = 0
        W = 1
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action != np.argmax(Q[state]):
                break
            W = W * 1 / 0.25
    return Q

def compute_greedy_policy(Q_values):
    greedy_policy = {state: np.argmax(action_values) for state, action_values in Q_values.items()}
    return greedy_policy

def plot_policy(greedy_policy, walls, goal_state):
    # Initialize the policy matrix with a value that's out of the action range
    policy_matrix = np.full((11, 11), -1)  # Default value for non-visited states

    # Update the matrix with the greedy policy actions
    for state, action in greedy_policy.items():
        policy_matrix[state] = action if state not in walls else -1
    
    # Mark the walls with a unique value that's not in the action range
    for wall in walls:
        policy_matrix[wall] = -2  # Wall indicator

    # Mark the goal state with a unique value that's not in the action range
    policy_matrix[goal_state] = 4  # Goal indicator

    # Create a color map
    cmap = plt.cm.viridis
    cmap.set_under('hot')  # Color for walls
    cmap.set_over('gold')  # Color for the goal state

    # Plot the policy matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(policy_matrix, cmap=cmap, origin='lower', vmin=-1.5, vmax=4.5)
    
    # Set the ticks and labels for both axes to represent the states
    ax.set_xticks(np.arange(policy_matrix.shape[1]))
    ax.set_yticks(np.arange(policy_matrix.shape[0]))
    ax.set_xticklabels(np.arange(policy_matrix.shape[1]))
    ax.set_yticklabels(np.arange(policy_matrix.shape[0]))
    
    # Draw gridlines
    ax.grid(which='major', color='m', linestyle='-', linewidth=1.5)
    
    # Color bar
    cbar = plt.colorbar(cax, ticks=range(4), orientation='vertical', shrink=0.8)
    cbar.ax.set_yticklabels(['LEFT', 'DOWN', 'RIGHT', 'UP'])
    cbar.set_label('Actions')
    plt.title('Greedy Policy Visualization')
    plt.show()


walls = [
    (0, 5), (2, 5), (3, 5), (4, 5),
    (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10),
    (6, 4), (7, 4), (9, 4), (10, 4),
]
goal_state = (10, 10)

if __name__ == "__main__":
    episodes_random = generate_episodes(random_policy, walls, goal_state, 10000)
    Q_values = off_policy_mc_prediction(episodes_random, gamma=1.0)
    # Print the estimated Q-values
    for state, values in Q_values.items():
        print(f"State: {state}, Q-values: {values}")
    greedy_policy = compute_greedy_policy(Q_values)
    plot_policy(greedy_policy, walls, goal_state)  # Arguments as Pass walls and goal_state 

