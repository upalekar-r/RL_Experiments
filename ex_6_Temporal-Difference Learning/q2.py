import numpy as np
import matplotlib.pyplot as plt

# # Parameters
# states = ['Term_A','A', 'B', 'C', 'D', 'E', 'Term_E']  # States
# episodes = 100  # Total episodes
# runs = 100  # Total runs for averaging
# actual_values = np.array([0,1/6, 2/6, 3/6, 4/6, 5/6,0])  # True probabilities

# # Function to simulate an episode
# def simulate_episode():
#     initial_state = 3  # Index of 'C'
#     states_visited = [initial_state]
#     rewards = [0]
    
#     # Simulating until termination
#     current_state = initial_state
#     while current_state > 0 and current_state < len(states) - 1:
#         if np.random.rand() < 0.5:
#             current_state -= 1  # Move left
#         else:
#             current_state += 1  # Move right
#         states_visited.append(current_state)
#         # Reward is 1 if termination on the right, otherwise 0
#         rewards.append(1 if current_state == len(states) - 1 else 0)
    
#     return states_visited, rewards

# # TD(0) learning for procducing graph 1
# # TD(0) learning function modified to return values at specified episodes
# def TD_learning(alpha, REQepisodes):
#     V = np.ones(len(states)) * 0.5  # Initial value estimates, intermediate value V(s) = 0.5 for all s
#     V[0], V[-1] = 0, 0  # Terminal states
#     episode_values = []  # To store the estimates at specified episodes
    
#     for episode in range(1, max(REQepisodes) + 1):
#         states_visited, rewards = simulate_episode()
#         for i in range(len(states_visited) - 1):
#             state = states_visited[i]
#             reward = rewards[i + 1]
#             next_state = states_visited[i + 1]
#             V[state] += alpha * (reward + V[next_state] - V[state])
#         if episode in REQepisodes:
#             episode_values.append(V.copy())
#     return episode_values

# # Run TD(0) and capture the values at specified episodes
# episode_values = TD_learning(alpha = 0.1, REQepisodes = [1, 10, 100])

# h = episode_values[0]
# c = episode_values[1]
# v = episode_values[2]
# # print(f'{h[1:6]}')
# # print(f'{c[1:6]}')
# # print(f'{v[1:6]}')


# # Plotting of Graph 1
# counting_states = ['A', 'B', 'C', 'D', 'E']  # Counting States
# initial_value = np.ones(5) * 0.5

# plt.plot(counting_states, initial_value, marker='o', linestyle='--', label=f'Episode {0}')
# plt.plot(counting_states, h[1:6], marker='o', linestyle='--', label=f'Episode {1}')
# plt.plot(counting_states, c[1:6], marker='o', linestyle='--', label=f'Episode {10}')
# plt.plot(counting_states, v[1:6], marker='o', linestyle='--', label=f'Episode {100}')

# plt.plot(counting_states, actual_values[1:6], marker='x', linestyle='--', color='k', label='True values')
# plt.xlabel('State')
# plt.ylabel('Estimated Value')
# plt.title('Estimated Values at Different Episodes for TD(0)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Constant-alpha MC learning, this can be used to make code efficient
# def MC_learning(alpha, episodes):
#     V = np.ones(len(states)) * 0.5  # Initial value estimates
#     V[0], V[-1] = 0, 0  # Terminal states
#     for _ in range(episodes):
#         states_visited, rewards = simulate_episode()
#         G = 0  # Initialize return
#         for i in reversed(range(1, len(states_visited))):
#             G = rewards[i] + G
#             state = states_visited[i-1]
#             V[state] += alpha * (G - V[state])
#     return V

# # Adjusted RMS error calculation to compare only the non-terminal states
# def rms_error(estimates):
#     return np.sqrt(np.mean((actual_values[1:-1] - estimates) ** 2))

# # For producing graph 2
# # Run simulations (Main loop adjusted for error calculation)
# TD_errs_1 = np.zeros(episodes)
# TD_errs_2 = np.zeros(episodes)
# TD_errs_3 = np.zeros(episodes)

# MC_errs_1 = np.zeros(episodes)
# MC_errs_2 = np.zeros(episodes)
# MC_errs_3 = np.zeros(episodes)
# MC_errs_4 = np.zeros(episodes)

# for _ in range(runs):
#     TD_V_1 = np.ones(len(states)) * 0.5  # Reset the V values 
#     TD_V_2 = np.ones(len(states)) * 0.5  # Reset the V values 
#     TD_V_3 = np.ones(len(states)) * 0.5  # Reset the V values 
    
#     TD_V_1[0], TD_V_1[-1] = 0, 0  # Terminal states are always 0
#     TD_V_2[0], TD_V_2[-1] = 0, 0  # Terminal states are always 0
#     TD_V_3[0], TD_V_3[-1] = 0, 0  # Terminal states are always 0

#     MC_V_1 = np.ones(len(states)) * 0.5  # Reset the V values 
#     MC_V_2 = np.ones(len(states)) * 0.5  # Reset the V values 
#     MC_V_3 = np.ones(len(states)) * 0.5  # Reset the V values 
#     MC_V_4 = np.ones(len(states)) * 0.5  # Reset the V values 

#     MC_V_1[0], MC_V_1[-1] = 0, 0  # Terminal states always be 0
#     MC_V_2[0], MC_V_2[-1] = 0, 0  # Terminal states always be 0
#     MC_V_3[0], MC_V_3[-1] = 0, 0  # Terminal states always be 0
#     MC_V_4[0], MC_V_4[-1] = 0, 0  # Terminal states always be 0

# # Generating TD errs for different alpha

#     for i in range(1, episodes + 1):
#         # Simulate and update values for a single episode for TD
#         alpha = 0.15
#         states_visited, rewards = simulate_episode()
#         for j in range(len(states_visited) - 1):
#             state = states_visited[j]
#             reward = rewards[j + 1]
#             next_state = states_visited[j + 1]
#             TD_V_1[state] += alpha * (reward + TD_V_1[next_state] - TD_V_1[state])
#         TD_errs_1[i-1] += rms_error(TD_V_1[1:-1])

#     for m in range(1, episodes + 1):
#         # Simulate and update values for a single episode for TD
#         alpha = 0.1
#         states_visited, rewards = simulate_episode()
#         for k in range(len(states_visited) - 1):
#             state = states_visited[k]
#             reward = rewards[k + 1]
#             next_state = states_visited[k + 1]
#             TD_V_2[state] += alpha * (reward + TD_V_2[next_state] - TD_V_2[state])
#         TD_errs_2[m-1] += rms_error(TD_V_2[1:-1])


#     for n in range(1, episodes + 1):
#         # Simulate and update values for a single episode for TD
#         alpha = 0.05
#         states_visited, rewards = simulate_episode()
#         for j in range(len(states_visited) - 1):
#             state = states_visited[j]
#             reward = rewards[j + 1]
#             next_state = states_visited[j + 1]
#             TD_V_3[state] += alpha * (reward + TD_V_3[next_state] - TD_V_3[state])
#         TD_errs_3[n-1] += rms_error(TD_V_3[1:-1])

#  # Update MC values after each episode for different alpha, we can also used MC_learning function for efficiency

#     for i in range(1, episodes + 1):
#         # Simulate and update values for a single episode for TD
#         alpha = 0.01
#         states_visited, rewards = simulate_episode()
#         G = 0  # Initialize return
#         for k in reversed(range(1, len(states_visited))):
#             G = rewards[k] + G
#             state = states_visited[k-1]
#             MC_V_1[state] += alpha * (G - MC_V_1[state])
#         MC_errs_1[i-1] += rms_error(MC_V_1[1:-1])
#         # Calculate and accumulate the RMS errs after adjustments

#     for i in range(1, episodes + 1):
#         # Simulate and update values for a single episode for TD
#         alpha = 0.02
#         states_visited, rewards = simulate_episode()
#         G = 0  # Initialize return
#         for k in reversed(range(1, len(states_visited))):
#             G = rewards[k] + G
#             state = states_visited[k-1]
#             MC_V_2[state] += alpha * (G - MC_V_2[state])
#         MC_errs_2[i-1] += rms_error(MC_V_2[1:-1])

#     for i in range(1, episodes + 1):
#         # Simulate and update values for a single episode for TD
#         alpha = 0.03
#         states_visited, rewards = simulate_episode()
#         G = 0  # Initialize return
#         for k in reversed(range(1, len(states_visited))):
#             G = rewards[k] + G
#             state = states_visited[k-1]
#             MC_V_3[state] += alpha * (G - MC_V_3[state])
#         MC_errs_3[i-1] += rms_error(MC_V_3[1:-1])

#     for i in range(1, episodes + 1):
#         # Simulate and update values for a single episode for TD
#         alpha = 0.04
#         states_visited, rewards = simulate_episode()
#         G = 0  # Initialize return
#         for k in reversed(range(1, len(states_visited))):
#             G = rewards[k] + G
#             state = states_visited[k-1]
#             MC_V_4[state] += alpha * (G - MC_V_4[state])
#         MC_errs_4[i-1] += rms_error(MC_V_4[1:-1])

# # Normalize errors by the number of runs
# average_td_errors = [TD_errs_1 / runs, TD_errs_2 / runs, TD_errs_3 / runs]
# average_mc_errors = [MC_errs_1 / runs, MC_errs_2 / runs, MC_errs_3 / runs, MC_errs_4 / runs]

# # Define alpha levels for clarity
# alphas_td = [0.15, 0.1, 0.05]
# alphas_mc = [0.01, 0.02, 0.03, 0.04]

# # Create a more descriptive plot with updated style
# plt.figure(figsize=(10, 6))
# for i, errors in enumerate(average_td_errors):
#     plt.plot(errors, linestyle='-', label=f'Temporal Difference α={alphas_td[i]}')

# for i, errors in enumerate(average_mc_errors):
#     plt.plot(errors, linestyle='-.', label=f'Monte Carlo α={alphas_mc[i]}')

# plt.xlabel('Episodes', fontsize=14)
# plt.ylabel('Root Mean Squared Error', fontsize=14)
# plt.title('Performance Comparison: Temporal Difference vs. Monte Carlo', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)  # lighter grid lines
# plt.tight_layout()
# plt.savefig('TD_vs_MC_Performance.png')  # Save the figure as a .png file
# plt.show()

def simulate_random_walk(num_states, start_state, left_reward, right_reward):
    """
    Simulates a random walk in an environment with a specified number of states,
    starting from a given state, and with specified rewards for reaching the left
    and right ends.
    """
    states_visited = [start_state]
    rewards = [0]  # Initial reward is 0 as we start from the middle.
    
    current_state = start_state
    while current_state > 0 and current_state < num_states - 1:
        if np.random.rand() < 0.5:
            current_state -= 1  # Move left
        else:
            current_state += 1  # Move right
        states_visited.append(current_state)
        # Assign rewards based on terminal state
        if current_state == 0:
            rewards.append(left_reward)
        elif current_state == num_states - 1:
            rewards.append(right_reward)
        else:
            rewards.append(0)
    
    return states_visited, rewards

def n_step_td(num_states, episodes, n, alpha, left_reward, right_reward):
    """
    Performs n-step TD learning for a given number of states, episodes,
    step size n, learning rate alpha, and reward structure.
    """
    V = np.zeros(num_states)  # Initialize state values
    V[0], V[-1] = left_reward, right_reward  # Set terminal state values
    
    for episode in range(episodes):
        states_visited, rewards = simulate_random_walk(num_states, num_states // 2, left_reward, right_reward)
        T = len(states_visited)  # Terminal time step
        for t in range(T):
            tau = t - n + 1  # Time step being updated
            if tau >= 0:
                G = sum([rewards[tau+k+1] * (0.9 ** k) for k in range(min(n, T-tau-1))])
                if tau + n < T:
                    G += (0.9 ** n) * V[states_visited[tau+n]]
                V[states_visited[tau]] += alpha * (G - V[states_visited[tau]])
    
    return V

# Parameters for the simulations
num_states = 50
episodes = 10  # We are interested in the first 10 episodes
n_values = [1, 2, 4, 8, 16, 32, 64]  # Different n values to test
alpha_values = np.linspace(0, 1, 100)  # Alphas to test
runs = 100  # Total runs for averaging
true_values = np.linspace(-1, 1, num_states)  # True state values for the 19-state random walk

# Function to calculate the RMS error given estimated values and true values
def rms_error(V, true_values):
    return np.sqrt(np.mean((V - true_values) ** 2))

# Function to perform n-step TD learning
def n_step_td_learning(num_states, episodes, n, alpha, start_state, left_reward, right_reward, true_values):
    V = np.zeros(num_states)  # Initialize state values
    V[0], V[-1] = left_reward, right_reward  # Set terminal state values
    rms_errors = []  # To store the RMS errors

    for episode in range(episodes):
        states_visited, rewards = simulate_random_walk(num_states, start_state, left_reward, right_reward)
        T = len(states_visited)
        for t in range(T):
            tau = t - n + 1  # State to be updated
            if tau >= 0:
                G = sum([rewards[tau+k+1] for k in range(min(n, T-tau-1))])
                if tau + n < T:
                    G += V[states_visited[tau+n]]
                V[states_visited[tau]] += alpha * (G - V[states_visited[tau]])
        rms_errors.append(rms_error(V[1:-1], true_values[1:-1]))  # Exclude terminal states

    return np.mean(rms_errors)  # Return the average RMS error over all episodes

# Run simulations for each combination of n and alpha
errors = np.zeros((len(n_values), len(alpha_values)))

for i, n in enumerate(n_values):
    for j, alpha in enumerate(alpha_values):
        error_sum = 0
        for run in range(runs):
            error = n_step_td_learning(num_states, episodes, n, alpha, num_states // 2, -1, 1, true_values)
            error_sum += error
        errors[i, j] = error_sum / runs

# Plotting the results
plt.figure(figsize=(12, 6))
for i, n in enumerate(n_values):
    plt.plot(alpha_values, errors[i, :], label='n={}'.format(n))

plt.title('Average RMS error over 19 states and first 10 episodes')
plt.xlabel('α')
plt.ylabel('Average RMS error')
plt.legend()
plt.show()