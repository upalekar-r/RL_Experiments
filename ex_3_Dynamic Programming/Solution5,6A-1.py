from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("white")

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]
class TruncatedPoissonDistribution:
    def __init__(self, mu: float, max_value: int):
        self.mu = mu
        self.max_value = max_value
        self.pmf_values = self._precompute_pmf()

    def _precompute_pmf(self) -> np.ndarray:
        pmf_values = np.zeros(self.max_value + 1)
        normalizer = 1 - np.exp(-self.mu)
        for k in range(self.max_value + 1):
            pmf_values[k] = (np.exp(-self.mu) * self.mu ** k) / (np.math.factorial(k) * normalizer)
        return pmf_values

    def pmf(self, k: int) -> float:
        return self.pmf_values[k]


class Gridworld5x5:
    """5x5 Gridworld"""

    def __init__(self) -> None:
        """
        State: (x, y) coordinates

        Actions: See class(Action).
        """
        self.rows = 5
        self.column = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.column)
        ]
        self.action_space = len(Action)

        # TODO set the locations of A and B, the next locations, and their rewards
        self.A = (0, 1)
        self.A_prime = (4, 1)
        self.A_reward = 10
        self.B = (0, 3)
        self.B_prime = (2, 3)
        self.B_reward = 5

    def transitions(
        self, state: Tuple, action: Action
    ) -> Tuple[Tuple[int, int], float]:
        """Get transitions from given (state, action) pair.

        Note that this is the 4-argument transition version p(s',r|s,a).
        This particular environment has deterministic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            subsequent_state: Tuple[int, int]
            reward: float
        """

        subsequent_state = tuple(map(lambda i, j: i + j, actions_to_dxdy(action), state))
        if state == self.A:
            subsequent_state = self.A_prime
            reward = self.A_reward

        elif state == self.B:
            subsequent_state = self.B_prime
            reward = self.B_reward

        # Else, check if the next step is within boundaries and return next state and reward
        elif subsequent_state not in self.state_space:
            subsequent_state = state
            reward = -1

        else:
            reward = 0

        return subsequent_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        subsequent_state, reward = self.transitions(state, action)
        # TODO compute the expected return
        ret = reward + gamma * V[subsequent_state]

        return ret


def iterative_policy_evaluation(grid_size=(5, 5), discount_factor=0.9, theta=1e-3):
    """
    Performs iterative policy evaluation on a 5x5 grid world.

    Returns:
    - V: Numpy array representing the value function for the grid.
    """
    delta = float('inf')
    w = Gridworld5x5()
    V = np.zeros(grid_size, dtype=float)
    action_prob = 1.0 / w.action_space

    while delta > theta:
        delta = 0

        for state in w.state_space:
            previous_value = V[state]
            expected_value = 0

            for action in range(w.action_space):
                expected_value += action_prob * w.expected_return(V, state, Action(action), discount_factor)

            V[state] = expected_value
            delta = max(delta, np.abs(previous_value - V[state]))

    return V

def num_to_string(p):
    # Dictionary to map number to its corresponding string direction
    direction_map = {
        0: 'up',
        1: 'left',
        2: 'down',
        3: 'right'
    }
    
    # Create a 5x5 matrix of objects
    result = np.zeros((5, 5), dtype=object)
    
    # Convert numbers to their respective string directions using vectorized operations
    for num, direction in direction_map.items():
        result[p == num] = direction

    return result
    
def value_iteration(grid_size=(5, 5), discount_factor=0.9, theta=1e-3):
    """
    Performs value iteration on a 5x5 grid world.

    Returns:
    - V: Numpy array representing the value function for the grid.
    - policy_string: A 5x5 numpy array containing string representations of the optimal actions.
    """
    delta = float('inf')
    w = Gridworld5x5()
    V = np.zeros(grid_size, dtype=float)
    policy = np.zeros(grid_size, dtype=float)

    while delta > theta:
        delta = 0

        for state in w.state_space:
            previous_value = V[state]
            values_for_actions = np.array([
                w.expected_return(V, state, Action(action), discount_factor)
                for action in range(w.action_space)
            ])

            V[state] = np.max(values_for_actions)
            policy[state] = np.argmax(values_for_actions)
            delta = max(delta, np.abs(previous_value - V[state]))

    policy_string = num_to_string(policy)

    return V, policy_string



def policy_iteration(grid_size=(5, 5), discount_factor=0.9, theta=1e-3):
    """
    Performs policy iteration on a 5x5 grid world.

    Returns:
    - V: Numpy array representing the value function for the grid.
    - policy_string: A 5x5 numpy array containing string representations of the optimal actions.
    """
    w = Gridworld5x5()
    V = np.zeros(grid_size, dtype=float)
    policy = np.zeros(grid_size, dtype=float)

    def policy_evaluation(current_policy):
        while True:
            delta = 0
            for state in w.state_space:
                previous_value = V[state]
                action = current_policy[state]
                V[state] = w.expected_return(V, state, action, discount_factor)
                delta = max(delta, np.abs(previous_value - V[state]))
            if delta < theta:
                break
        return V

    def policy_improvement():
        is_policy_stable = True
        previous_policy = policy.copy()
        for state in w.state_space:
            action_values = np.array([
                w.expected_return(V, state, Action(action), discount_factor)
                for action in range(w.action_space)
            ])
            best_action = np.argmax(action_values)
            policy[state] = best_action
            V[state] = action_values[best_action]
            if previous_policy[state] != policy[state]:
                is_policy_stable = False
        return is_policy_stable

    while True:
        V = policy_evaluation(policy)
        policy_stable = policy_improvement()
        if policy_stable:
            break

    policy_string = num_to_string(policy)
    return V, policy_string



class TruncatedPoissonDistribution(object):
    def __init__(self, mean, threshold):
        assert isinstance(mean, int), "should be greater than 0"
        assert 0 < threshold < 1.0, "threshold should be between 0 and 1"

        self.mean = mean
        self.truncated_threshold = threshold
        self.truncated_values, self.truncated_prob = self.truncate_poisson()
    
    def sample(self):
        return np.random.choice(a=self.truncated_values, p=self.truncated_prob)


    def truncate_poisson(self):
        distribution = poisson(self.mean)
        max_k = 0
        while distribution.pmf(max_k) > self.truncated_threshold:
            max_k += 1
        value_list = list(np.linspace(start=0, stop=max_k, num=max_k+1, dtype=int))
        prob_list = [distribution.pmf(k) for k in value_list]
        prob_list = (prob_list / np.sum(prob_list)).tolist()

        return value_list, prob_list
    
    def __iter__(self):
        return zip(self.truncated_values, self.truncated_prob)


def compute_expected_return(state, action, env, gamma, state_value):
    car_num_to_move = env.move_car(state, action)
    state_after_move = [state[0] - car_num_to_move, state[1] + car_num_to_move]
    recent_v = 0
    for n_1 in range(21):
        for n_2 in range(21):
            prob = env.p_lot1[state_after_move[0]][n_1] * env.p_lot2[state_after_move[1]][n_2]
            reward = env.compute_reward(moved_cars=car_num_to_move,
                                        car_num_lot1=state_after_move[0],
                                        car_num_lot2=state_after_move[1])
            recent_v += (prob * (reward + (gamma * state_value[n_1,n_2])))
            
    return recent_v

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

   
    def step(self, s, a):
        car_num_lot1, car_num_lot2 = s
        move_car_num = self.move_car(s, a)
        car_num_lot1_after_move,  car_num_lot2_after_move = car_num_lot1 - move_car_num, car_num_lot2 + move_car_num
        request_car_lot1_num, request_car_lot2_num = self.request_distribution_lot1.sample(), self.request_distribution_lot2.sample()
        return_car_lot1_num, return_car_lot2_num = self.return_distribution_lot1.sample(), self.return_distribution_lot2.sample()
        car_num_lot1_recent = self.update_car_num(car_num_lot1_after_move, request_car_lot1_num, return_car_lot1_num)
        car_num_lot2_recent = self.update_car_num(car_num_lot2_after_move, request_car_lot2_num, return_car_lot2_num)
        prob = self.p_lot1[car_num_lot1_after_move][car_num_lot1_recent] * \
            self.p_lot2[car_num_lot2_after_move][car_num_lot2_recent]
        reward = self.compute_reward(move_car_num, car_num_lot1_after_move, car_num_lot2_after_move)
        subsequent_state = [car_num_lot1_recent, car_num_lot2_recent]
        return subsequent_state, reward, prob
    
    def reset(self):
        pass
    
    @staticmethod
    def update_car_num(cars_num, request_cars_num, return_cars_num):
        res = min(max(cars_num - request_cars_num, 0) + return_cars_num, 20)
        return res

    def compute_reward(self, moved_cars, car_num_lot1, car_num_lot2):
        reward = -2 * abs(moved_cars) + self.r_lot1[car_num_lot1] + self.r_lot2[car_num_lot2]
        return reward

    @staticmethod
    def move_car(state, action):
        return int(np.clip(action, a_min=-state[1], a_max=state[0]))

    def compute_reward_modified(self, moved_cars, car_num_lot1, car_num_lot2):
        if moved_cars > 0:
            moved_cars = moved_cars - 1
        lot1_penalty = 0
                                         
        if car_num_lot1 > 10:
            lot1_penalty = -4
        
        lot2_penalty = 0

        if car_num_lot2 > 10:
            lot2_penalty = -4
            
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
    previous_policy = policy.copy()
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



def plot_results(results_list):
    for res in results_list:
        policy = np.flip(res['policy'], axis=0)
        car_num_lot1 = np.arange(20, -1, -1)
        car_num_lot2 = np.arange(21)

        fig, ax = plt.subplots()
        im = ax.imshow(policy)

        ax.set_xticks(np.arange(len(car_num_lot2)))
        ax.set_yticks(np.arange(len(car_num_lot1)))
        ax.set_xticklabels(car_num_lot2)
        ax.set_yticklabels(car_num_lot1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(car_num_lot1)):
            for j in range(len(car_num_lot2)):
                text = ax.text(j, i, policy[i, j], ha="center", va="center", color="w")
        ax.set_title(res['title'])
        fig.tight_layout()
        plt.show()

    x, y = np.meshgrid(np.arange(21), np.arange(21))
    z = results_list[0]['state_value']

    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, cmap='coolwarm', edgecolor='none')
    ax.set_title('State Value')
    plt.show()


if __name__ == '__main__':
    Question5a = iterative_policy_evaluation()
    print ('Answer5a V(s) - ')
    print(Question5a)

    Question5b, pi_s = value_iteration()
    print('Answer5b V(s) - ')
    print(Question5b)
    print('Pi(s) - ')
    print(pi_s)

    Question5c, pi_s = policy_iteration()
    print('Answer5c V(s) - ')
    print(Question5c)
    print('Pi(s) - ')
    print(pi_s)
    
    env = JackCarRental()
    threshold = 1e-3
    gamma = 0.9
    results_list = run_policy_iteration(env, threshold, gamma)
    plot_results(results_list)
