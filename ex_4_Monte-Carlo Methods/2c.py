import numpy as np
import matplotlib.pyplot as plt

# Define the simulation of the MDP as per Example 5.5
def simulate_episode(behavior_policy_prob, gamma, max_steps=10000):
    # This simulates an episode using the behavior policy
    # behavior_policy_prob is the probability of taking 'left' according to the behavior policy
    steps = 0
    while True:
        steps += 1
        if np.random.rand() < 0.9:  # 90% chance to loop back to s
            if np.random.rand() > behavior_policy_prob:  # Taken 'right' action
                return 0, steps  # No reward, episode ends
        else:  # 10% chance to move to the terminal state with reward +1
            return 1, steps  # Reward +1, episode ends

        if steps >= max_steps:
            return 0, steps  # Capping the steps to avoid infinite loop

# Set parameters for the simulation
gamma = 1.0  # Discount factor
behavior_policy_prob = 0.5  # 50% chance to go left according to behavior policy
num_episodes = 1000000  # Total number of episodes to simulate
runs = 10  # Number of runs

# Data structure to keep track of estimates
value_estimates = np.zeros((runs, num_episodes))

# Perform the runs
for run in range(runs):
    for episode in range(num_episodes):
        reward, steps = simulate_episode(behavior_policy_prob, gamma)
        value = reward * (gamma ** (steps - 1))  # Calculate discounted return
        value_estimates[run, episode] = value if episode == 0 else \
            (value_estimates[run, episode - 1] * episode + value) / (episode + 1)  # Incremental average

# Plotting
plt.figure(figsize=(10, 6))
episodes = np.arange(1, num_episodes + 1)
for run in range(runs):
    plt.plot(episodes, value_estimates[run], label=f'Run {run + 1}')

plt.xscale('log')
plt.xlabel('Episodes (log scale)')
plt.ylabel('Monte-Carlo estimate of $V_π(s)$ with ordinary importance sampling')
plt.title('Monte-Carlo estimate of $V_π(s)$ with ordinary importance sampling (ten runs)')
plt.legend()
plt.show()

