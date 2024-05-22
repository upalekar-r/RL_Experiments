import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import Enum
import random

class Action(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]

def reset() -> Tuple[int, int]:
    return (0, 0)

def is_valid_move(state: Tuple[int, int], next_state: Tuple[int, int], walls) -> bool:
    if next_state in walls or not (0 <= next_state[0] < 11 and 0 <= next_state[1] < 11):
        print(f"Invalid move attempted from {state}.")
        return False
    return True

def simulate(state: Tuple[int, int], action: Action) -> Tuple[Tuple[int, int], int]:
    walls = [
        (0, 5), (2, 5), (3, 5), (4, 5), (5, 0), (5, 2), (5, 3), (5, 4),
        (5, 5), (5, 6), (5, 7), (5, 9), (5, 10), (6, 4), (7, 4), (9, 4), (10, 4)
    ]
    goal_state = (10, 10)
    
    dxdy = actions_to_dxdy(action)
    next_state = (state[0] + dxdy[0], state[1] + dxdy[1])

    if not is_valid_move(state, next_state, walls):
        return state, 0

    reward = 1 if next_state == goal_state else 0
    if next_state == goal_state:
        print("Goal state achieved (10, 10) and state reset.")
        next_state = reset()

    return next_state, reward

def random_policy(state: Tuple[int, int]) -> Action:
    return random.choice(list(Action))

def agent(steps: int, trials: int, policy: Callable[[Tuple[int, int]], Action]):
    goal_state = (10, 10)
    cumulative_rewards = np.zeros((trials, steps))

    for t in range(trials):
        state = reset()
        total_reward = 0

        for i in range(steps):
            action = policy(state)
            next_state, reward = simulate(state, action)
            total_reward += reward
            cumulative_rewards[t, i] = total_reward
            state = next_state

    return cumulative_rewards

def plot_cumulative_rewards(cumulative_rewards):
    plt.figure(figsize=(10, 6))
    for trial_rewards in cumulative_rewards:
        plt.plot(trial_rewards, linestyle='--')
    plt.plot(cumulative_rewards.mean(axis=0), color='black', linewidth=2)
    plt.title('Cumulative Reward per Trial')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.show()

# if __name__ == "__main__":
#     trials = 10
#     steps = 10000
#     cumulative_rewards = agent(steps, trials, random_policy)
#     plot_cumulative_rewards(cumulative_rewards)

def worse_policy(state: Tuple[int, int]) -> Action:
    # Implementation of a worse policy
    # Example: Prefer moving LEFT or DOWN
    if state[0] > 0:
        return Action.LEFT
    elif state[1] > 0:
        return Action.DOWN
    else:
        return random.choice(list(Action))

def better_policy(state: Tuple[int, int]) -> Action:
    # Implementation of a better policy
    # Example: Prefer moving RIGHT or UP towards the goal
    # goal_state = (10, 10)
    # if state[1] < goal_state[1]:
    #     return Action.UP
    # if state[0] < goal_state[0]:
    #     return Action.RIGHT
    # else:
    #     return random.choice(list(Action))
    return random.choices(list(Action), weights = (10,10,40,40))[0]

goal_state = (random.randint (0,10), random.randint(0,10))

def learned_policy(state: Tuple[int,int]):
    #random_policy(goal_state)
    #r = (goal_state[0]/(goal_state[0] + goal_state[1])) * 100
    #u = (goal_state[1]/(goal_state[0] + goal_state[1])) * 100
    c1 = goal_state[0] - state[0]
    c2 = goal_state[1] - state[1]
    if c1 !=0 and c2!=0:
        prob = [0.75,0.76,0.75,0.76]
    if c1 ==0:
        prob = [0.75,0.75,0.75,0.8]
    if c2 ==0:
        prob = [0.75,0.75,0.8,0.75]
    if c1>=0 and c2>=0:
        prob = [0.75,0.75,0.76,0.76]
    if c1>=0 and c2<=0:
        prob = [0.75,0.76,0.76,0.75]
    if c1<=0 and c2>=0:
        prob = [0.76,0.75,0.75,0.76]
    if c1<=0 and c2<=0:
        prob = [0.76,0.76,0.75,0.75]
 
    return random.choices(list(Action),prob)[0]
    

def plot_cumulative_rewards(cumulative_rewards_dict):
    plt.figure(figsize=(10, 6))

    for label, cumulative_rewards in cumulative_rewards_dict.items():
        # Plot each trial's reward with low opacity
        for trial_rewards in cumulative_rewards:
            plt.plot(trial_rewards, linestyle='--', alpha=0.5)
        # Plot the mean reward with solid line
        plt.plot(cumulative_rewards.mean(axis=0), linewidth=2, label=label)

    plt.title('Cumulative Reward per Trial')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

def main():
    trials = 10
    steps = 10000

    cumulative_rewards_random = agent(steps, trials, random_policy)
    cumulative_rewards_worse = agent(steps, trials, worse_policy)
    cumulative_rewards_better = agent(steps, trials, better_policy)
    cumulative_rewards_learned = agent(steps, trials, learned_policy)

    cumulative_rewards_dict = {
        "Random Policy": cumulative_rewards_random,
        "Worse Policy": cumulative_rewards_worse,
        "Better Policy": cumulative_rewards_better,
        "learned Policy": cumulative_rewards_learned
    }

    plot_cumulative_rewards(cumulative_rewards_dict)

if __name__ == "__main__":
    main()




    
