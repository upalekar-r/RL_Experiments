import numpy as np
import matplotlib.pyplot as plt

# Parameters
states = ['Term_A','A', 'B', 'C', 'D', 'E', 'Term_E']  # States
episodes = 100  # Total episodes
runs = 100  # Total runs for averaging
actual_values = np.array([0,1/6, 2/6, 3/6, 4/6, 5/6,0])  # True probabilities

# Function to simulate an episode
def simulate_episode():
    initial_state = 3  # Index of 'C'
    states_visited = [initial_state]
    rewards = [0]
    
    # Simulating until termination
    current_state = initial_state
    while current_state > 0 and current_state < len(states) - 1:
        if np.random.rand() < 0.5:
            current_state -= 1  # Move left
        else:
            current_state += 1  # Move right
        states_visited.append(current_state)
        # Reward is 1 if termination on the right, otherwise 0
        rewards.append(1 if current_state == len(states) - 1 else 0)
    
    return states_visited, rewards

# TD(0) learning for procducing graph 1
# TD(0) learning function modified to return values at specified episodes
def TD_learning(alpha, REQepisodes):
    V = np.ones(len(states)) * 0.5  # Initial value estimates, intermediate value V(s) = 0.5 for all s
    V[0], V[-1] = 0, 0  # Terminal states
    episode_values = []  # To store the estimates at specified episodes
    
    for episode in range(1, max(REQepisodes) + 1):
        states_visited, rewards = simulate_episode()
        for i in range(len(states_visited) - 1):
            state = states_visited[i]
            reward = rewards[i + 1]
            next_state = states_visited[i + 1]
            V[state] += alpha * (reward + V[next_state] - V[state])
        if episode in REQepisodes:
            episode_values.append(V.copy())
    return episode_values

# Run TD(0) and capture the values at specified episodes
episode_values = TD_learning(alpha = 0.1, REQepisodes = [1, 10, 100])

h = episode_values[0]
c = episode_values[1]
v = episode_values[2]
# print(f'{h[1:6]}')
# print(f'{c[1:6]}')
# print(f'{v[1:6]}')


# Plotting of Graph 1
counting_states = ['A', 'B', 'C', 'D', 'E']  # Counting States
initial_value = np.ones(5) * 0.5

plt.plot(counting_states, initial_value, marker='o', linestyle='--', label=f'Episode {0}')
plt.plot(counting_states, h[1:6], marker='o', linestyle='--', label=f'Episode {1}')
plt.plot(counting_states, c[1:6], marker='o', linestyle='--', label=f'Episode {10}')
plt.plot(counting_states, v[1:6], marker='o', linestyle='--', label=f'Episode {100}')

plt.plot(counting_states, actual_values[1:6], marker='x', linestyle='--', color='k', label='True values')
plt.xlabel('State')
plt.ylabel('Estimated Value')
plt.title('Estimated Values at Different Episodes for TD(0)')
plt.legend()
plt.grid(True)
plt.show()

# Constant-alpha MC learning, this can be used to make code efficient
def MC_learning(alpha, episodes):
    V = np.ones(len(states)) * 0.5  # Initial value estimates
    V[0], V[-1] = 0, 0  # Terminal states
    for _ in range(episodes):
        states_visited, rewards = simulate_episode()
        G = 0  # Initialize return
        for i in reversed(range(1, len(states_visited))):
            G = rewards[i] + G
            state = states_visited[i-1]
            V[state] += alpha * (G - V[state])
    return V

# Adjusted RMS error calculation to compare only the non-terminal states
def rms_error(estimates):
    return np.sqrt(np.mean((actual_values[1:-1] - estimates) ** 2))

# For producing graph 2
# Run simulations (Main loop adjusted for error calculation)
TD_errs_1 = np.zeros(episodes)
TD_errs_2 = np.zeros(episodes)
TD_errs_3 = np.zeros(episodes)

MC_errs_1 = np.zeros(episodes)
MC_errs_2 = np.zeros(episodes)
MC_errs_3 = np.zeros(episodes)
MC_errs_4 = np.zeros(episodes)

for _ in range(runs):
    TD_V_1 = np.ones(len(states)) * 0.5  # Reset the V values 
    TD_V_2 = np.ones(len(states)) * 0.5  # Reset the V values 
    TD_V_3 = np.ones(len(states)) * 0.5  # Reset the V values 
    
    TD_V_1[0], TD_V_1[-1] = 0, 0  # Terminal states are always 0
    TD_V_2[0], TD_V_2[-1] = 0, 0  # Terminal states are always 0
    TD_V_3[0], TD_V_3[-1] = 0, 0  # Terminal states are always 0

    MC_V_1 = np.ones(len(states)) * 0.5  # Reset the V values 
    MC_V_2 = np.ones(len(states)) * 0.5  # Reset the V values 
    MC_V_3 = np.ones(len(states)) * 0.5  # Reset the V values 
    MC_V_4 = np.ones(len(states)) * 0.5  # Reset the V values 

    MC_V_1[0], MC_V_1[-1] = 0, 0  # Terminal states always be 0
    MC_V_2[0], MC_V_2[-1] = 0, 0  # Terminal states always be 0
    MC_V_3[0], MC_V_3[-1] = 0, 0  # Terminal states always be 0
    MC_V_4[0], MC_V_4[-1] = 0, 0  # Terminal states always be 0

# Generating TD errs for different alpha

    for i in range(1, episodes + 1):
        # Simulate and update values for a single episode for TD
        alpha = 0.15
        states_visited, rewards = simulate_episode()
        for j in range(len(states_visited) - 1):
            state = states_visited[j]
            reward = rewards[j + 1]
            next_state = states_visited[j + 1]
            TD_V_1[state] += alpha * (reward + TD_V_1[next_state] - TD_V_1[state])
        TD_errs_1[i-1] += rms_error(TD_V_1[1:-1])

    for m in range(1, episodes + 1):
        # Simulate and update values for a single episode for TD
        alpha = 0.1
        states_visited, rewards = simulate_episode()
        for k in range(len(states_visited) - 1):
            state = states_visited[k]
            reward = rewards[k + 1]
            next_state = states_visited[k + 1]
            TD_V_2[state] += alpha * (reward + TD_V_2[next_state] - TD_V_2[state])
        TD_errs_2[m-1] += rms_error(TD_V_2[1:-1])


    for n in range(1, episodes + 1):
        # Simulate and update values for a single episode for TD
        alpha = 0.05
        states_visited, rewards = simulate_episode()
        for j in range(len(states_visited) - 1):
            state = states_visited[j]
            reward = rewards[j + 1]
            next_state = states_visited[j + 1]
            TD_V_3[state] += alpha * (reward + TD_V_3[next_state] - TD_V_3[state])
        TD_errs_3[n-1] += rms_error(TD_V_3[1:-1])

 # Update MC values after each episode for different alpha, we can also used MC_learning function for efficiency

    for i in range(1, episodes + 1):
        # Simulate and update values for a single episode for TD
        alpha = 0.01
        states_visited, rewards = simulate_episode()
        G = 0  # Initialize return
        for k in reversed(range(1, len(states_visited))):
            G = rewards[k] + G
            state = states_visited[k-1]
            MC_V_1[state] += alpha * (G - MC_V_1[state])
        MC_errs_1[i-1] += rms_error(MC_V_1[1:-1])
        # Calculate and accumulate the RMS errs after adjustments

    for i in range(1, episodes + 1):
        # Simulate and update values for a single episode for TD
        alpha = 0.02
        states_visited, rewards = simulate_episode()
        G = 0  # Initialize return
        for k in reversed(range(1, len(states_visited))):
            G = rewards[k] + G
            state = states_visited[k-1]
            MC_V_2[state] += alpha * (G - MC_V_2[state])
        MC_errs_2[i-1] += rms_error(MC_V_2[1:-1])

    for i in range(1, episodes + 1):
        # Simulate and update values for a single episode for TD
        alpha = 0.03
        states_visited, rewards = simulate_episode()
        G = 0  # Initialize return
        for k in reversed(range(1, len(states_visited))):
            G = rewards[k] + G
            state = states_visited[k-1]
            MC_V_3[state] += alpha * (G - MC_V_3[state])
        MC_errs_3[i-1] += rms_error(MC_V_3[1:-1])

    for i in range(1, episodes + 1):
        # Simulate and update values for a single episode for TD
        alpha = 0.04
        states_visited, rewards = simulate_episode()
        G = 0  # Initialize return
        for k in reversed(range(1, len(states_visited))):
            G = rewards[k] + G
            state = states_visited[k-1]
            MC_V_4[state] += alpha * (G - MC_V_4[state])
        MC_errs_4[i-1] += rms_error(MC_V_4[1:-1])

# Normalize errors by the number of runs
average_td_errors = [TD_errs_1 / runs, TD_errs_2 / runs, TD_errs_3 / runs]
average_mc_errors = [MC_errs_1 / runs, MC_errs_2 / runs, MC_errs_3 / runs, MC_errs_4 / runs]

# Define alpha levels for clarity
alphas_td = [0.15, 0.1, 0.05]
alphas_mc = [0.01, 0.02, 0.03, 0.04]

# Create a more descriptive plot with updated style
plt.figure(figsize=(10, 6))
for i, errors in enumerate(average_td_errors):
    plt.plot(errors, linestyle='-', label=f'Temporal Difference α={alphas_td[i]}')

for i, errors in enumerate(average_mc_errors):
    plt.plot(errors, linestyle='-.', label=f'Monte Carlo α={alphas_mc[i]}')

plt.xlabel('Episodes', fontsize=14)
plt.ylabel('Root Mean Squared Error', fontsize=14)
plt.title('Performance Comparison: Temporal Difference vs. Monte Carlo', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)  # lighter grid lines
plt.tight_layout()
plt.savefig('TD_vs_MC_Performance.png')  # Save the figure as a .png file
plt.show()
