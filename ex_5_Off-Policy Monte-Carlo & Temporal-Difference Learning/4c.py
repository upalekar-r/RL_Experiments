import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Actions represented as vectors
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)] # Right, Down, Left, Up
ACTION_NAMES = ['RIGHT', 'DOWN', 'LEFT', 'UP']

# Grid and walls setup
GRID_SIZE = 11
WALLS = [(0, 5), (2, 5), (3, 5), (4, 5), (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10), (6, 4), (7, 4), (9, 4), (10, 4)]
GOAL_STATE = (10, 10)

def step(state, action):
    """Take a step in the environment."""
    if state in WALLS:
        return state, -1  # Penalize for hitting a wall
    next_state = tuple(np.add(state, action))
    if next_state in WALLS or not (0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE):
        return state, -1  # Stay in place if hit a wall or out of bounds, with penalty
    return next_state, 0 if next_state == GOAL_STATE else -1

def generate_episode(policy, Q, epsilon, num_steps=100):
    """Generate episodes."""
    state = (0, 0)
    episode = []
    for _ in range(num_steps):
        if state == GOAL_STATE:
            break
        action = policy(state, Q, epsilon)
        next_state, reward = step(state, ACTIONS[action])
        episode.append((state, action, reward))
        state = next_state
    return episode

def epsilon_greedy_policy(state, Q, epsilon):
    """Epsilon-greedy policy."""
    if np.random.rand() < epsilon or state not in Q:
        return np.random.choice(len(ACTIONS))
    return np.argmax(Q[state])

def off_policy_mc_prediction(episodes, gamma=0.9):
    """Off-policy MC prediction."""
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    C = defaultdict(lambda: np.zeros(len(ACTIONS)))
    for episode in episodes:
        G = 0
        W = 1.0
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action != np.argmax(Q[state]):
                break
            W *= 1.0/0.25  # Assuming behavior policy is uniform
    return Q

def plot_value_function(Q):
    """Plot the value function."""
    V = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    plt.imshow(V, cmap='hot')
    plt.colorbar(label='Value')
    plt.title('Value Function')
    plt.show()

# Generating episodes and estimating Q-values
num_episodes = 10000
epsilon = 0.1

# Random policy episodes
random_episodes = [generate_episode(lambda s, Q, epsilon: np.random.randint(len(ACTIONS)), {}, epsilon, 100) for _ in range(num_episodes)]
Q_random = off_policy_mc_prediction(random_episodes)

# Epsilon-greedy policy episodes
Q_initial = defaultdict(lambda: np.zeros(len(ACTIONS)))
epsilon_greedy_episodes = [generate_episode(epsilon_greedy_policy, Q_initial, epsilon, 100) for _ in range(num_episodes)]
Q_epsilon_greedy = off_policy_mc_prediction(epsilon_greedy_episodes)

# Plotting
plot_value_function(Q_random)
plot_value_function(Q_epsilon_greedy)
