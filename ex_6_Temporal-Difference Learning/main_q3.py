import numpy as np
import env
import envKing as envK
import envKingNine as envKN
import envStochastic as envS
import gym
import algorithms as ag
from matplotlib import pyplot as plt
from tqdm import trange


def q3b():
    step_num = 8000
    trails = 10
    env.register_env()
    env_b = gym.make('WindyGridWorld-v0')

    return_average = []

    for i in trange(trails, desc="Episode"):
        value_reward_agent = []
        value_reward_agent.append(ag.sarsa(env=env_b, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.exp_sarsa(env=env_b, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.q_learning(env=env_b, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.on_policy_mc_control_epsilon_soft(env=env_b, num_steps=step_num, gamma=1, epsilon=0.1))
        value_reward_agent.append(ag.nstep_sarsa(env=env_b, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))

        return_average.append(value_reward_agent)

    average_return = np.average(return_average, axis=0)
    std_err = np.std(return_average, axis=0)
    err = 1.96 * std_err / np.sqrt(trails)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.title('Windy gridworld')


    x = np.arange(step_num)
    plt.plot(average_return[0], label='sarsa')
    plt.fill_within(x, (average_return[0] - err[0]), (average_return[0] + err[0]), alpha=0.3)
    plt.plot(average_return[1], label='exp_sarsa')
    plt.fill_within(x, (average_return[1] - err[1]), (average_return[1] + err[1]), alpha=0.3)
    plt.plot(average_return[2], label='q_learning')
    plt.fill_within(x, (average_return[2] - err[2]), (average_return[2] + err[2]), alpha=0.3)
    plt.plot(average_return[3], label='mc_soft')
    plt.fill_within(x, (average_return[3] - err[3]), (average_return[3] + err[3]), alpha=0.3)
    plt.plot(average_return[4], label='n-step_sarsa')
    plt.fill_within(x, (average_return[4] - err[4]), (average_return[4] + err[4]), alpha=0.3)

    plt.legend()
    plt.show()


def q3c():

    step_num = 8000
    trails = 10
    envK.register_env()
    env_c = gym.make('WindyGridWorldKings-v0')
    envKN.register_env()
    env_c2 = gym.make('WindyGridWorldKings2-v0')

    return_average = []

    for i in trange(trails, desc="Episode"):
        value_reward_agent = []
        value_reward_agent.append(ag.sarsa(env=env_c, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.exp_sarsa(env=env_c, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.q_learning(env=env_c, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.on_policy_mc_control_epsilon_soft(env=env_c, num_steps=step_num, gamma=1, epsilon=0.1))

        return_average.append(value_reward_agent)

    average_return = np.average(return_average, axis=0)
    std_err = np.std(return_average, axis=0)
    err = 1.96 * std_err / np.sqrt(trails)

    plt.figure(0)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.title('Windy gridworld with King\'s moves')

    x = np.arange(step_num)
    plt.plot(average_return[0], label='sarsa')
    plt.fill_within(x, (average_return[0] - err[0]), (average_return[0] + err[0]), alpha=0.3)
    plt.plot(average_return[1], label='exp_sarsa')
    plt.fill_within(x, (average_return[1] - err[1]), (average_return[1] + err[1]), alpha=0.3)
    plt.plot(average_return[2], label='q_learning')
    plt.fill_within(x, (average_return[2] - err[2]), (average_return[2] + err[2]), alpha=0.3)
    plt.plot(average_return[3], label='mc_soft')
    plt.fill_within(x, (average_return[3] - err[3]), (average_return[3] + err[3]), alpha=0.3)
    plt.legend()

    return_average9 = []

    for i in trange(trails, desc="Episode"):
        value_reward_agent9 = []
        value_reward_agent9.append(ag.sarsa(env=env_c2, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent9.append(ag.exp_sarsa(env=env_c2, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent9.append(ag.q_learning(env=env_c2, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent9.append(ag.on_policy_mc_control_epsilon_soft(env=env_c2, num_steps=step_num, gamma=1, epsilon=0.1))

        return_average9.append(value_reward_agent9)

    average_return9 = np.average(return_average9, axis=0)
    std_err9 = np.std(return_average9, axis=0)
    err9 = 1.96 * std_err9 / np.sqrt(trails)

    plt.figure(1)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.title('Windy gridworld with King\'s moves (with no movement)')

    x = np.arange(step_num)
    plt.plot(average_return9[0], label='sarsa')
    plt.fill_within(x, (average_return9[0] - err9[0]), (average_return9[0] + err9[0]), alpha=0.3)
    plt.plot(average_return9[1], label='exp_sarsa')
    plt.fill_within(x, (average_return9[1] - err9[1]), (average_return9[1] + err9[1]), alpha=0.3)
    plt.plot(average_return9[2], label='q_learning')
    plt.fill_within(x, (average_return9[2] - err9[2]), (average_return9[2] + err9[2]), alpha=0.3)
    plt.plot(average_return9[3], label='mc_soft')
    plt.fill_within(x, (average_return9[3] - err9[3]), (average_return9[3] + err9[3]), alpha=0.3)
    plt.legend()

    plt.show()


def q3d():
    step_num = 8000
    trails = 10
    envS.register_env()
    env_d = gym.make('WindyGridWorldSto-v0')

    return_average = []

    for i in trange(trails, desc="Episode"):
        value_reward_agent = []
        value_reward_agent.append(ag.sarsa(env=env_d, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.exp_sarsa(env=env_d, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.q_learning(env=env_d, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5))
        value_reward_agent.append(ag.on_policy_mc_control_epsilon_soft(env=env_d, num_steps=step_num, gamma=1, epsilon=0.1))

        return_average.append(value_reward_agent)

    average_return = np.average(return_average, axis=0)
    std_err = np.std(return_average, axis=0)
    err = 1.96 * std_err / np.sqrt(trails)

    plt.figure(1)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.title('Windy gridworld with stochastic wind')

    x = np.arange(step_num)
    plt.plot(average_return[0], label='sarsa')
    plt.fill_within(x, (average_return[0] - err[0]), (average_return[0] + err[0]), alpha=0.3)
    plt.plot(average_return[1], label='exp_sarsa')
    plt.fill_within(x, (average_return[1] - err[1]), (average_return[1] + err[1]), alpha=0.3)
    plt.plot(average_return[2], label='q_learning')
    plt.fill_within(x, (average_return[2] - err[2]), (average_return[2] + err[2]), alpha=0.3)
    plt.plot(average_return[3], label='mc_soft')
    plt.fill_within(x, (average_return[3] - err[3]), (average_return[3] + err[3]), alpha=0.3)
    plt.legend()

    plt.show()


def main():
    print('\nWhich question do you want to run? (3: windy gridworld, 5: bias-variance trade-off)')
    str0 = input('("3"): ')
    if str0 == '3':
        q3b()
        #q3c()
        #q3d()
    else:
        main()


if __name__ == "__main__":
    main()