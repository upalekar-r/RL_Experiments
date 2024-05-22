import numpy as np
from scipy.stats import poisson
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from copy import copy
import seaborn as sns
sns.set_style("white")

class TruncatedPoissonDistribution(object):
    def __init__(self, mean, threshold):
        assert isinstance(mean, int), mean > 0
        assert 0 < threshold < 1.0

        self.mean = mean
        self.truncated_threshold = threshold
        self.truncated_values, self.truncated_prob = self.truncate_poisson()

    def truncate_poisson(self):
        distribution = poisson(self.mean)
        max_k = 0
        while distribution.pmf(max_k) > self.truncated_threshold:
            max_k += 1
        value_list = list(np.linspace(start=0, stop=max_k, num=max_k+1, dtype=int))
        prob_list = [distribution.pmf(k) for k in value_list]
        prob_list = (prob_list / np.sum(prob_list)).tolist()

        return value_list, prob_list

    def sample(self):
        return np.random.choice(a=self.truncated_values, p=self.truncated_prob)

    def __iter__(self):
        return zip(self.truncated_values, self.truncated_prob)


class JackCarRental(object):
    def __init__(self, max_car_num=20):
        self.max_n = max_car_num
        self.state_value = np.zeros((self.max_n + 1, self.max_n + 1))
        self.action_space = np.linspace(start=-5, stop=5, num=11, dtype=int)
        self.truncated_threshold = 1e-6
        self.request_distribution_lot1 = TruncatedPoissonDistribution(3, self.truncated_threshold)
        self.return_distribution_lot1 = TruncatedPoissonDistribution(3, self.truncated_threshold)
        self.request_distribution_lot2 = TruncatedPoissonDistribution(4, self.truncated_threshold)
        self.return_distribution_lot2 = TruncatedPoissonDistribution(2, self.truncated_threshold)
        self.p_lot1, self.r_lot1 = self.open_to_close(self.request_distribution_lot1, self.return_distribution_lot1)
        self.p_lot2, self.r_lot2 = self.open_to_close(self.request_distribution_lot2, self.return_distribution_lot2)

    def reset(self):
        pass

    def step(self, s, a):
        car_num_lot1, car_num_lot2 = s
        move_car_num = self.move_car(s, a)
        car_num_lot1_after_move = car_num_lot1 - move_car_num
        car_num_lot2_after_move = car_num_lot2 + move_car_num
        request_car_lot1_num = self.request_distribution_lot1.sample()
        request_car_lot2_num = self.request_distribution_lot2.sample()
        return_car_lot1_num = self.return_distribution_lot1.sample()
        return_car_lot2_num = self.return_distribution_lot2.sample()
        car_num_lot1_recent = self.update_car_num(car_num_lot1_after_move, request_car_lot1_num, return_car_lot1_num)
        car_num_lot2_recent = self.update_car_num(car_num_lot2_after_move, request_car_lot2_num, return_car_lot2_num)
        prob = self.p_lot1[car_num_lot1_after_move][car_num_lot1_recent] * \
            self.p_lot2[car_num_lot2_after_move][car_num_lot2_recent]
        reward = self.compute_reward(move_car_num, car_num_lot1_after_move, car_num_lot2_after_move)
        subsequent_state = [car_num_lot1_recent, car_num_lot2_recent]
        return subsequent_state, reward, prob

    def compute_reward(self, moved_cars, car_num_lot1, car_num_lot2):
        reward = -2 * abs(moved_cars) + self.r_lot1[car_num_lot1] + self.r_lot2[car_num_lot2]
        return reward

    def compute_reward_modified(self, moved_cars, car_num_lot1, car_num_lot2):
        if moved_cars > 0:
            moved_cars = moved_cars - 1
        lot1_penalty = 0

        lot2_penalty = 0

        if car_num_lot2 > 10:
            lot2_penalty = -4                                 
        if car_num_lot1 > 10:
            lot1_penalty = -4
        

            
        reward =  (-2 * abs(moved_cars)) + self.r_lot1[car_num_lot1] + self.r_lot2[car_num_lot2] + lot1_penalty + lot2_penalty
        return reward

    def open_to_close(self, request_distribution, return_distribution):
        p_arr = np.zeros((26, 21))
        r_arr = np.zeros(26)

        for request_num, request_prob in request_distribution:
            for n in range(26):
                r_arr[n] += request_prob * 10 * min(n, request_num)

            for return_num, return_prob in return_distribution:
                for n in range(26):
                    recent_n = self.update_car_num(n, request_num, return_num)
                    p_arr[n][recent_n] += request_prob * return_prob

        return p_arr, r_arr

    @staticmethod
    def update_car_num(cars_num, request_cars_num, return_cars_num):
        res = min(max(cars_num - request_cars_num, 0) + return_cars_num, 20)
        return res

    @staticmethod
    def move_car(state, action):
        #Compute the actual number of cars to be moved from lot1 to lot2
        return int(np.clip(action, a_min=-state[1], a_max=state[0]))
    
    



def compute_expected_return(state, action, env, gamma, state_value):
    car_num_to_move = env.move_car(state, action)
    state_after_move = [state[0] - car_num_to_move, state[1] + car_num_to_move]
    recent_v = 0
    for n_1 in range(21):
        for n_2 in range(21):
            prob = env.p_lot1[state_after_move[0]][n_1] * env.p_lot2[state_after_move[1]][n_2]
            reward = env.compute_reward_modified(moved_cars=car_num_to_move,
                                                 car_num_lot1=state_after_move[0],
                                                 car_num_lot2=state_after_move[1])
            recent_v += (prob * (reward + (gamma * state_value[n_1,n_2])))
            
    return recent_v

def policy_evaluation(state_value, policy, env, threshold, gamma):
    iter_counter = 0
    while True:
        iter_counter += 1
        is_terminal = True
        recent_state_value = state_value.copy()
        for i in range(21):
            for j in range(21):
                old_v = state_value[i,j]
                action = policy[i,j]
                recent_v = compute_expected_return([i,j], action, env, gamma, state_value)
                delta = abs(recent_v-old_v)
                
                if delta > threshold:
                    is_terminal = False
                recent_state_value[i, j] = recent_v
        state_value = recent_state_value.copy()
        if is_terminal:
            break

    return state_value

def policy_improvement(state_value, policy, env, gamma):
    is_stable = True
    old_policy = policy.copy()
    for i in range(21):
        for j in range(21):
            old_a = policy[i,j]
            highest_val = -float("inf")
            for action in range(-5,6):
                val = compute_expected_return([i,j], action, env, gamma, state_value)
                action_array = []
                if val > highest_val:
                    recent_a = action
                    highest_val = val
            val_0 = compute_expected_return([i,j], 0, env, gamma, state_value)
            if val_0 == highest_val:
                recent_a = 0
            if old_a != recent_a:
                is_stable = False
            policy[i, j] = recent_a

    return policy, is_stable

def run_policy_iteration(env, threshold, gamma):
    policy = np.zeros((21, 21), dtype=int)
    state_value = np.zeros((21, 21))
    policy_iter_counter = 0
    results_list = []
    while True:
        state_value = policy_evaluation(state_value, policy, env, threshold, gamma)
        policy, is_stable = policy_improvement(state_value, policy, env, gamma)
        results_list.append({"state_value": state_value.copy(),
                             "policy": policy.copy(),
                             "title": f"Iteration = {policy_iter_counter}"})
        if is_stable:
            break
        else:
            policy_iter_counter += 1       
    return results_list

if __name__ == '__main__':
    env = JackCarRental()
    env.reset()
    threshold = 1e-3
    gamma = 0.9
    results_list = run_policy_iteration(env, threshold, gamma)
    for res in results_list: 
        policy = np.flip(res['policy'], axis=0)

        car_num_lot1 = list(np.linspace(start=20, stop=0, num=21, dtype=int))
        car_num_lot2 = list(np.linspace(start=0, stop=20, num=21, dtype=int))

        fig, ax = plt.subplots()
        im = ax.imshow(policy)

        ax.set_xticks(np.arange(len(car_num_lot2)))
        ax.set_yticks(np.arange(len(car_num_lot1)))
        ax.set_xticklabels(car_num_lot2)
        ax.set_yticklabels(car_num_lot1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
        for i in range(len(car_num_lot1)):
            for j in range(len(car_num_lot2)):
                text = ax.text(j, i, policy[i, j], ha="center", va="center", color="w")

        ax.set_title(res['title'])
        fig.tight_layout()
        plt.show()
  
    x = np.linspace(0, 20, 21)
    y = np.linspace(0, 20, 21)

    X, Y = np.meshgrid(x, y)
    Z =  results_list[0]['state_value']
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='coolwarm', edgecolor='none')
    ax.set_title('State Value')
    plt.show()