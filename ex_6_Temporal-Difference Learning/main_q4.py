from algorithms import on_policy_mc_control_epsilon_soft, sarsa, exp_sarsa, nstep_sarsa, q_learning
import gym
import numpy as np
from env import register_env
import matplotlib.pyplot as plt

plt.style.use('Solarize_Light2')
def Run(env,text):
    #MC
    trials = 10
    num_time_steps = 10000
    total_ep = np.zeros([10, 10000])
    for trial in range(10):
        r, Q, policy, total_ep[trial][:] = on_policy_mc_control_epsilon_soft(env=env, total_steps=10000,gamma=1, epsilon=0.1,num_episodes=100000)
        
    avg_reward = np.average(total_ep, axis=0)
    plt.plot(avg_reward, label = "Monte Carlo")
    std_error_rewards = np.std(total_ep,axis=0)/np.sqrt(trials)
    c_upper = avg_reward + std_error_rewards * 1.96
    c_lower = avg_reward - std_error_rewards * 1.96

    steps_x = np.arange(num_time_steps)
    plt.fill_between(steps_x, c_lower, c_upper, alpha=0.5)


    #SARSA
    total_ep = np.zeros([10, 10000])
    for trial in range(10):
        total_ep[trial][:] = sarsa(env=env, num_steps=10000,gamma=1, step_size=0.5,epsilon=0.1,total_episodes=100000)
        

    avg_reward = np.average(total_ep, axis=0)
    plt.plot(avg_reward, label = "SARSA")
    std_error_rewards = np.std(total_ep,axis=0)/np.sqrt(trials)
    c_upper = avg_reward + std_error_rewards * 1.96
    c_lower = avg_reward - std_error_rewards * 1.96

    steps_x = np.arange(num_time_steps)
    plt.fill_between(steps_x, c_lower, c_upper, alpha=0.5)


    #Expected SARSA
    total_ep = np.zeros([10, 10000])
    for trial in range(10):
        total_ep[trial][:] = exp_sarsa(env=env, num_steps=10000, gamma=1, epsilon=0.1, step_size= 0.5)
    
    avg_reward = np.average(total_ep, axis=0)
    plt.plot(avg_reward, label = "Exp SARSA")
    std_error_rewards = np.std(total_ep,axis=0)/np.sqrt(trials)
    c_upper = avg_reward + std_error_rewards * 1.96
    c_lower = avg_reward - std_error_rewards * 1.96

    steps_x = np.arange(num_time_steps)
    plt.fill_between(steps_x, c_lower, c_upper, alpha=0.5)

    #N-Step SARSA
    total_ep = np.zeros([trials, 10000])
    for trial in range(trials):
        total_ep[trial][:] = nstep_sarsa(env=env, num_steps=10000, gamma=0.9, epsilon=0.1, step_size=0.5, n=4)
    
    avg_reward = np.average(total_ep, axis=0)
    plt.plot(avg_reward, label = "n-step SARSA")
    std_error_rewards = np.std(total_ep,axis=0)/np.sqrt(trials)
    c_upper = avg_reward + std_error_rewards * 1.96
    c_lower = avg_reward - std_error_rewards * 1.96

    steps_x = np.arange(num_time_steps)
    plt.fill_between(steps_x, c_lower, c_upper, alpha=0.5)


    #Q-Learning
    total_ep = np.zeros([10, 10000])
    for trial in range(10):
        total_ep[trial][:] = q_learning(env=env, num_steps=10000, gamma=1, epsilon=0.1, step_size= 0.5)
    
    avg_reward = np.average(total_ep, axis=0)
    plt.plot(avg_reward, label = "Q-Learning")
    std_error_rewards = np.std(total_ep,axis=0)/np.sqrt(trials)
    c_upper = avg_reward + std_error_rewards * 1.96
    c_lower = avg_reward - std_error_rewards * 1.96

    steps_x = np.arange(num_time_steps)
    plt.fill_between(steps_x, c_lower, c_upper, alpha=0.5)
    plt.title(text)
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.show()


def main():
    # #Q4a
    # register_env()
    # env = gym.make('WindyGridWorld-v0', types = 0)
    # Run(env=env, text="Windy Grid-World")

    # #Q4b
    # register_env()
    # env = gym.make('WindyGridWorld-v0', types = 1)
    # Run(env=env, text="Windy Grid-World (King's Move)")

    # #Q4c
    # register_env()
    # env = gym.make('WindyGridWorld-v0', types = 2)
    # Run(env=env, text="Windy Grid-World (Stochastic)")

    #Q5a
    register_env()
    env = gym.make('WindyGridWorld-v0', types = 0)

    training = [1,20,50]

    for N in training:
        
        plt.figure()
        Q = sarsa(env=env, num_steps=10000000,gamma=1, step_size=0.5,epsilon=0.1,total_episodes=N)
        
        learning_targets = sarsa(env=env, num_steps=10000000,gamma=1, step_size=0.5,epsilon=0.1,total_episodes=500)

        plt.subplot(3,1,1)
        plt.title("Learning targets N = " + str(N))
        plt.hist(learning_targets,bins=50)


        
        Q,_,_,_ = on_policy_mc_control_epsilon_soft(env=env, total_steps=10000000,gamma=0.99, epsilon=0.1,num_episodes=N)
        
        learning_targets,_,_,_ = on_policy_mc_control_epsilon_soft(env=env, total_steps=10000000,gamma=0.99,epsilon=0.1,num_episodes=500)
        
        plt.subplot(3,1,2)
        # plt.title("Learning targets N = " + str(N))
        plt.hist(learning_targets,bins=50)

        

        Q = nstep_sarsa(env=env, num_steps=10000000, gamma=0.9, epsilon=0.1, step_size=0.5, n=4)
        
        learning_targets = nstep_sarsa(env=env, num_steps=10000000, gamma=0.9, epsilon=0.1, step_size=0.5, n=4)
        
        plt.subplot(3,1,3)
        # plt.title("Learning targets N = " + str(N))
        plt.hist(learning_targets,bins=50)

        
        
    
    plt.show()

if __name__ == "__main__":
    main()

