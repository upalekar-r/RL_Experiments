import env
import Algorithms
import gym
import numpy as np
from tqdm import Trialange
from tiles3 import tiles
import matplotlib.pyplot as plt

def Q3b ():
    # Initialize the environment
    Env = gym.make('FourRooms-v0')
    
    # Run the semi-gradient SARSA algorithm
    Trial = Algorithms.semi_grad_sarsa(env=Env, trials=100, eps=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=1)
    y = np.average(Trial, axis=0)
    x = list(range(Trial.shape[1]))
    y_std = np.std(Trial, 0)
    
    # Determine the confidence interval
    l = y - 1.96 * y_std / np.sqrt(Trial.shape[0])
    h = y + 1.96 * y_std / np.sqrt(Trial.shape[0])
    
    # Plotting the results
    plt.figure()
    plt.plot(x, y, label='sarsas', color='tab:orange')
    plt.fill_between(x, l, h, color='tab:orange', alpha=0.2)
    
    plt.xlabel("Episodes")
    plt.ylabel("Steps per Episode")
    plt.title("Performance of Semi-Gradient SARSA")
    plt.legend()
    plt.show()


def Q3c():
    env.register_env()
    Env = gym.make('FourRooms-v0')

# Conduct trials with varying levels of aggregation
    Trial1 = Algorithms.semi_grad_sarsa(env=Env, trials=100, eps=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=1)
    Trial2 = Algorithms.semi_grad_sarsa(env=Env, trials=100, eps=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=2)
    Trial3 = Algorithms.semi_grad_sarsa(env=Env, trials=100, eps=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=3)
    Trial4 = Algorithms.semi_grad_sarsa(env=Env, trials=100, eps=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=4)

# Prepare data for plotting
    x1 = list(range(Trial1.shape[1]))
    x2 = list(range(Trial2.shape[1]))
    x3 = list(range(Trial3.shape[1]))
    x4 = list(range(Trial4.shape[1]))

    y1 = np.average(Trial1, axis=0)
    y2 = np.average(Trial2, axis=0)
    y3 = np.average(Trial3, axis=0)
    y4 = np.average(Trial4, axis=0)

    y1_std = np.std(Trial1, 0)
    y2_std = np.std(Trial2, 0)
    y3_std = np.std(Trial3, 0)
    y4_std = np.std(Trial4, 0)

    l1 = y1 - 1.96 * y1_std / np.sqrt(Trial1.shape[0])
    l2 = y2 - 1.96 * y2_std / np.sqrt(Trial2.shape[0])
    l3 = y3 - 1.96 * y3_std / np.sqrt(Trial3.shape[0])
    l4 = y4 - 1.96 * y4_std / np.sqrt(Trial4.shape[0])

    h1 = y1 + 1.96 * y1_std / np.sqrt(Trial1.shape[0])
    h2 = y2 + 1.96 * y2_std / np.sqrt(Trial2.shape[0])
    h3 = y3 + 1.96 * y3_std / np.sqrt(Trial3.shape[0])
    h4 = y4 + 1.96 * y4_std / np.sqrt(Trial4.shape[0])

 # Plotting results 
    plt.figure(1)
    plt.plot(x1, y1, label='sarsas-1', color='tab:orange')
    plt.fill_between(x1, l1, h1, alpha=0.2, color='tab:orange')
    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(x2, y2, label='sarsas-2', color='tab:orange')
    plt.fill_between(x2, l2, h2, alpha=0.2, color='tab:orange')
    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(x3, y3, label='sarsas-3', color='tab:orange')
    plt.fill_between(x3, l3, h3, alpha=0.2, color='tab:orange')
    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

    plt.figure(4)
    plt.plot(x4, y4, label='sarsas-4', color='tab:orange')
    plt.fill_between(x4, l4, h4, alpha=0.2, color='tab:orange')
    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

def Q3d():
    # Initialize the environment
    env.register_env()
    # Run the extended semi-gradient SARSA algorithm on the 'FourRooms-v0' environment
    # Collect trial data for 100 episodes with extended feature set
    Trial = Algorithms.semi_grad_sarsa_extend(env=gym.make('FourRooms-v0'), trials=100, eps=100, step_size=0.1, gamma=0.99, epsilon=0.1)
    y = np.average(Trial, axis=0)
    x = list(range(Trial.shape[1]))
    y_std = np.std(Trial, 0)
    # Calculate the 95% confidence interval for the average steps per episode
    l = y - 1.96 * y_std / np.sqrt(Trial.shape[0])
    h = y + 1.96 * y_std / np.sqrt(Trial.shape[0])

    plt.figure(2)
    plt.plot(x, y, label='extended features sarsa', color='tab:orange')
    plt.fill_between(x, l, h, alpha=0.2, color='tab:orange')

    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad SARSA extended features")
    plt.legend()
    plt.show()

def Q3e():
    env.register_env()

    Trial = Algorithms.semi_grad_sarsa_extend_more(env=gym.make('FourRooms-v0'), trials=100, eps=100, step_size=0.1, gamma=0.99, epsilon=0.1)
    y = np.average(Trial, axis=0)
    x = list(range(Trial.shape[1]))
    y_std = np.std(Trial, 0)
    l = y - 1.96 * y_std / np.sqrt(Trial.shape[0])
    h = y + 1.96 * y_std / np.sqrt(Trial.shape[0])

    plt.figure(2)
    plt.plot(x, y, label='extended features sarsa', color='tab:orange')
    plt.fill_between(x, l, h, alpha=0.2, color='tab:orange')

    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad SARSA extended more features")
    plt.legend()
    plt.show()

def main():
    Q3b()
    Q3c()
    Q3d()
    Q3e()

if __name__ == "__main__":
    main()

