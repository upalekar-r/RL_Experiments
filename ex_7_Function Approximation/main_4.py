import env
import Algorithms
import gym
import numpy as np
from tqdm import trange
from tiles3 import tiles
import matplotlib.pyplot as plt

def Q4(trails, episodes):
    # Initialize lists to store steps to completion for each trial across different step sizes
    Trial1 = []
    Trial2 = []
    Trial3 = []

    # Iterate over the number of trials using a progress bar (assumed trange is from tqdm for visual progress indication)
    for _ in trange(trails):
        # Run the mountain car problem with SARSA algorithm using three different step sizes
        _, _, st1 = Algorithms.mountain_car_sarsa(episodes, step_size=0.1 / 8, gamma=1)
        _, _, st2 = Algorithms.mountain_car_sarsa(episodes, step_size=0.2 / 8, gamma=1)
        _, _, st3 = Algorithms.mountain_car_sarsa(episodes, step_size=0.5 / 8, gamma=1)

        # Collect the steps per episode for each trial and step size
        Trial1.append(st1)
        Trial2.append(st2)
        Trial3.append(st3)

    Trial1_ave = np.average(Trial1, 0)
    Trial2_ave = np.average(Trial2, 0)
    Trial3_ave = np.average(Trial3, 0)

    # Plotting the learning curves for the mountain car problem with different step sizes
    plt.figure()
    plt.plot(Trial1_ave, label='step_size = 0.1/8')
    plt.plot(Trial2_ave, label='step_size = 0.2/8')
    plt.plot(Trial3_ave, label='step_size = 0.5/8')

    plt.title('Mountain car learning curves')
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    plt.show()

def Q4_3dview(episodes):
    # Execute the mountain car problem with the SARSA algorithm to obtain weights and tile coding helpers
    iht, w, _ = Algorithms.mountain_car_sarsa(episodes, step_size=0.1 / 8, gamma=1)

    # Initialize a grid to store Q-values for visualization
    z = np.zeros([40, 40])

    # Iterate over a discretized version of the state space
    for i in range(40):
        for j in range(40):
            # Consider each action's contribution to Q-values, but only store the last to visualize
            for a in [0, 1, 2]:
                # Compute state based on current indices, adjusting for discretization
                s = [-1.2 + (1.7/40) * i, -0.07 + (0.14/40) * j]
                # Use tile coding to get feature indices for this state-action pair
                T = tiles(iht, 8, [8 * s[0] / 1.7, 8 * s[1] / 0.14], [a])

                # Accumulate weights for the features active for this state-action pair
                q = 0
                for t in T:
                    q += w[t]
                Q = -q  # Negate Q to visualize (specific to this implementation's visualization choice)
            z[i, j] = Q  # Update grid with computed Q-value

    # Create meshgrid for plotting
    x = np.arange(z.shape[0])
    y = np.arange(z.shape[1])
    x, y = np.meshgrid(x, y)

    # Set up figure and 3D subplot
    fig = plt.figure(figsize=(14, 9))  # Specify figure size for better visualization
    ax = fig.add_subplot(111, projection='3d')
    # Create a surface plot using the coolwarm color map to represent Q-values
    surf = ax.plot_surface(x, y, z, cmap='coolwarm', edgecolor='none')
    # Add a color bar to help interpret Q-values based on color
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

    # Label axes and set title
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Mountain Car Performance over Episodes = 9000')

    # Display the plot
    plt.show()

def main():
    # Q4(trails=100, episodes=500)
    Q4_3dview(episodes=9000)

if __name__ == "__main__":
    main()