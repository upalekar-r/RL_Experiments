import gym
from policy import create_epsilon_policy
from typing import Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from tqdm import trange
from typing import Callable, Tuple


def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    # print(episode)
    return episode


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, total_steps: int, num_episodes: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(num_episodes)
    ep_list = []
    step = 0
    # print(num_episodes)
    for i in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        
        episode = generate_episode(env, policy, es=False)
        # trial_list.append(len(episode))
        G = 0
        for t in trange(len(episode) - 1, -1, -1, desc='step'):
            state,action,reward = episode[t]
            G = gamma*G + reward
            if (state,action) not in episode[:t]:
                N[state][action]+=1
                Q[state][action]=Q[state][action] + ((G - Q[state][action]))/(N[state][action])
            step+=1
            ep_list.append(i)
            if step == total_steps:               
                return returns, Q, policy, ep_list
        returns[i] = G
    return returns, Q, policy, ep_list


def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float, total_episodes):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    ep_list = []
    c = 0
    learning_targets

    for episodes in trange(total_episodes, desc='episodes'):
        state = env.reset()
        action = policy(state)
        while c<num_steps:
        
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            Q[state][action] = Q[state][action] + step_size * (reward + gamma*Q[next_state][next_action] - Q[state][action])
            action, state = next_action, next_state
            if done == True:
                break
                # episodes += 1
                # state = env.reset()
                # action = policy(state)
            c += 1
            ep_list.append(episodes)
       

    return(ep_list)


def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    n : int
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    max_episodes = 2000
    total_time_steps = 0
    ep_list = []

    for episode in trange(max_episodes, desc='episodes'):
        state = env.reset()
        c = 0
        T = np.inf
        action = policy(state)

        actions = [action]
        states = [state]
        rewards = [0]

        while True:
            if c < T:
                state, reward, done, _ = env.step(action)
                states.append(state)
                rewards.append(reward)

                if state == env.goal_pos:
                    T = c+1

                else:
                    action = policy(state)
                    actions.append(action)

            tau = c - n + 1

            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau+n+1, T+1)):
                    G += np.power(gamma, i - tau - 1) * rewards[i]

                if tau + n < T:

                    G += np.power(gamma, n) * Q[states[tau+n]][actions[tau+n]]

                Q[states[tau]][actions[tau]] += step_size*(G - Q[states[tau]][actions[tau]])

            if tau == T - 1:
                break
                
            c += 1
            total_time_steps += 1
            ep_list.append(episode)
            if total_time_steps == num_steps:
                return ep_list


    return ep_list
 
    


def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    e = 0
    ep_list = []
    state = env.reset()
    action = policy(state)


    for i in trange(num_steps, desc='steps'):
        # print(state, " ", action)
        next_state, reward, done, _ = env.step(action)
        next_action = policy(next_state)
        Q[state][action] = Q[state][action] + step_size * (reward + gamma*np.mean(Q[next_state][:]) - Q[state][action])
        action, state = next_action, next_state
        if done == True:
            e += 1
            state = env.reset()
            action = policy(state)
        ep_list.append(e)

    return ep_list


def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    e = 0
    ep_list = []
    state = env.reset()

    for i in trange(num_steps, desc='steps'):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        Q[state][action] = Q[state][action] + step_size * (reward + gamma*np.max(Q[next_state][:]) - Q[state][action])
        state = next_state
        if done == True:
            e += 1
            state = env.reset()
            
        ep_list.append(e)

    return(ep_list)


def td_prediction(env: gym.Env, gamma: float, episodes, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    # for step in range n:
    pass


def learning_targets(
    V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO

    pass
