from typing import Tuple, List, Set, Dict, Callable
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from tqdm import trange
import random 

def plot_performance(eps: List[float], clrs: List[str], G: np.ndarray):
    fig, plt1 = plt.subplots(1, figsize=(8, 8), facecolor="white")

    def subplot_performance(eps: List[float], clrs: List[str], x_data: List[int], G, s_plt):
        for i, param in enumerate(eps):
            y_data = np.average(G[i], axis=0)
            s_plt.plot(x_data,y_data, color=clrs[i], label="$\epsilon$ = " + str(param))
            y_stderr = np.std(G[i], axis=0)
            y_stderr *= 1 / math.sqrt(G.shape[1])
            y_stderr *= 1.96
            s_plt.fill_between(x_data, np.subtract(y_data, y_stderr), np.add(y_data, y_stderr), alpha=0.2, color=clrs[i])
        G_Max = np.amax(G)
        y_data = [G_Max for x in range(G.shape[2])]
        s_plt.plot(x_data,y_data, color="black", label="Upper Bound")
        return
    x_data = [x for x in range(G.shape[2])]
    subplot_performance(eps, clrs, x_data, G, plt1)
    plt1.legend(loc="lower right")
    plt1.set_xlabel("Episodes")
    plt1.set_ylabel("Discounted Return")
    plt1.set_title("Performance plot:" + " Episodes = " + str(G.shape[2]) + "," + " Trials = " + str(G.shape[1]))
    plt.show()

    return

def track0():
    return np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
      ], dtype=np.int32)

def track1():
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      ], dtype=np.int32)


class Racetrack(object):
    def __init__(self, version):
        if version == "v1":
            self.domain_arr = track0().copy()
        else:
            self.domain_arr = track1().copy()

        # domain size
        self.height, self.width = self.domain_arr.shape

        # State space consists of:
        # Agent location
        self.empty_cell_locs = self.render_cell_locations(val=0.0)
        self.track_cell_locs = self.render_cell_locations(val=1.0)
        self.start_cell_locs = self.render_cell_locations(val=2.0)
        self.finish_cell_locs = self.render_cell_locations(val=3.0)

        # Action space
        self.action_space = [[-1, -1], [-1, 0], [-1, 1],
                             [0, -1], [0, 0], [0, 1],
                             [1, -1], [1, 0], [1, 1]]

        # construct the state space
        self.state_space = []
        for loc in self.start_cell_locs + self.empty_cell_locs + self.finish_cell_locs:
            for i in range(5):
                for j in range(5):
                    self.state_space.append(loc + [i, j])

        # track the agent's location
        self.state = None
        self.action = None
        self.t = None

    def reset(self):
        start_location = random.sample(self.start_cell_locations, 1)[0]
        start_velocity = [0, 0]
        state = start_location + start_velocity
        reward = None
        done = False
        self.state = tuple(state)
        self.t = 0
        return state, reward, done

    def step(self, state, action):
        # reward is -1 for every time step until the agent passes the finish line
        reward = -1
        self.t += 1
        
        # with the probability = 0.1, set action = [0, 0]
        if np.random.random() < 0.1:
            action = [0, 0]

        # update the velocity components
        # note that, both velocity is discrete and constraint within [0, 4]
        next_velocity_x = np.clip(state[2] + action[0], a_min=0, a_max=4)
        next_velocity_y = np.clip(state[3] + action[1], a_min=0, a_max=4)
        next_state_velocity = [next_velocity_x, next_velocity_y]

        # only the cells on the start line can have both 0 velocities
        if next_state_velocity == [0, 0]:
            if state not in self.start_cell_locations:
                # non-zero for velocities
                if state[2] == 0 and state[3] != 0:
                    next_state_velocity = [0, 1]
                if state[2] != 0 and state[3] == 0:
                    next_state_velocity = [1, 0]
                if state[2] != 0 and state[3] != 0:
                    non_zero_idx = random.sample([0, 1], 1)[0]
                    next_state_velocity[non_zero_idx] = 1

        # update the next state ation based on the updated velocities
        next_state_location = [np.clip(state[0] + next_state_velocity[0], a_min=0, a_max=self.width-1),
                          np.clip(state[1] + next_state_velocity[1], a_min=0, a_max=self.height-1)]

        # check whether the agent hits the track
        if next_state_location in self.track_cell_locations:
            # move back to the start line
            next_state_location = random.sample(self.start_cell_locations, 1)[0]
            # reduce veity to be 0s
            next_state_velocity = [0, 0]
            # episode continue
            done = False
            # next state
            next_state = next_state_location + next_state_velocity
            return next_state, reward, done

        # check whether the agent pass the finish line
        if next_state_location in self.finish_cell_locations:
            next_state = next_state_location + next_state_velocity
            done = True
            return next_state, 0, done

        # otherwise combine the next state
        next_state = next_state_location + next_state_velocity
        # termination
        done = False

        # track the agent's state
        self.state = tuple(next_state)
        self.action = action
        return next_state, reward, done

    def render_cell_locations(self, val):
        row_location_indices, col_location_indices = np.where(self.domain_arr == val)
        cell_locations = [[c, (self.height-1) - r] for r, c in zip(row_location_indices, col_location_indices)]
        return cell_locations

    def render(self):
        plt.clf()
        plt.title(f"s = {self.state}, a = {self.action}")
        plot_arr = self.domain_arr.copy()
        plot_arr[(self.height - 1) - self.state[1], self.state[0]] = 4
        plt.imshow(plot_arr)
        plt.show(block=False)
        plt.pause(0.01)


# Q6a

def MCPolicySelector(Epsilon, bestAction):
    action_space = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
    if random.random() <= (((1 - Epsilon) + (Epsilon / 9))):
        return(bestAction)
    else:
        action_space.remove(bestAction)
        return(random.choice(action_space))


def FVMCRC(version, NumEpisodes, Epsilon, Gamma):
    env = Racetrack(version)
    #Arbitrary soft policy(just random in this case as it doesn't matter)
    trackDimOriginal = np.shape(env.domain_arr)
    trackDim = (trackDimOriginal[1],trackDimOriginal[0])
    Policy = np.zeros((trackDim[0],trackDim[1],5,5), dtype = object)
    for i in range(trackDim[0]):
        for j in range(trackDim[1]):
            for k in range(5):
                for l in range(5):
                    Policy[i][j][k][l] = random.choice(env.action_space)

    policy = Policy.copy()
    #Initiating Qvals
    Qval = np.zeros((trackDim[0],trackDim[1],5,5,9))
    for q in range(trackDim[0]):
        for w in range(trackDim[1]):
            for e in range(5):
                for r in range(5):
                    for t in range(9):
                        Qval[q][w][e][r][t] = 0
    #Initiating Returns
    Returns = np.zeros((trackDim[0],trackDim[1],5,5), dtype = object)
    for q in range(trackDim[0]):
        for w in range(trackDim[1]):
            for e in range(5):
                for r in range(5):
                    Returns[q][w][e][r] = [[],[],[],[],[],[],[],[],[]]
    #Episode Return
    G0 = np.zeros(NumEpisodes)
    #Episode loop
    for ep in trange(int(NumEpisodes)):
        #Generate an Episode
        state,reward,done = env.reset()
        episodeDeets = []
        while True:
            bestAction = Policy[state[0],state[1],state[2],state[3]]
            action = MCPolicySelector(Epsilon, bestAction)
            next_state, reward, done = env.step(state, action)
            episodeDeets.append((state,action,reward))
            if done == True:
                break
            state = next_state
        #initiate episode return
        G = 0
        #Visited states for first-visit MC control
        visited = []
        for j in reversed(episodeDeets):
            G = (Gamma*G) + j[2]
            CS = j[0]
            CA = j[1]
            if CS not in visited:
                visited.append(CS)
                CAindex = env.action_space.index(CA)
                Returns[CS[0],CS[1],CS[2],CS[3]][CAindex].append(G)
                Qval[CS[0],CS[1],CS[2],CS[3],CA] = np.average(Returns[CS[0],CS[1],CS[2],CS[3]][CAindex])
        G0[ep] = G
        for i in range(trackDim[0]):
            for j in range(trackDim[1]):
                for k in range(5):
                    for l in range(5):
                        bestAction = np.random.choice(np.where(Qval[i,j,k,l] == Qval[i,j,k,l].max())[0])
                        policy[i,j,k,l] = bestAction
    return G0

def MCRCpolicy(version, numTrials, numEpisodes, Epsilon, Gamma):
    G0total = []
    for trial in trange(numTrials, desc="Trial Progress", leave=False):
        G0 = FVMCRC(version, numEpisodes, Epsilon, Gamma)
        G0total.append(G0)

    return G0total

#Q6b
def MCPolicySelectorGreedy(Epsilon, bestAction):
    action_space = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
    if random.random() <= (((1 - Epsilon) + (Epsilon / 9))):
        return(bestAction)
    else:
        action_space.remove(bestAction)
        return(random.choice(action_space))

def OffPolicyMC(version, NumEpisodes, Epsilon, Gamma):
    env = Racetrack(version)
    trackDimOriginal = np.shape(env.domain_arr)
    trackDim = (trackDimOriginal[1],trackDimOriginal[0])
    #Initiating Qvals
    Qval = np.zeros((trackDim[0],trackDim[1],5,5,9))
    for q in range(trackDim[0]):
        for w in range(trackDim[1]):
            for e in range(5):
                for r in range(5):
                    for t in range(9):
                        Qval[q][w][e][r][t] = 0
    #Initiating Cvalues
    Cval = np.zeros((trackDim[0],trackDim[1],5,5,9))
    #Arg max policy
    Policy = np.zeros((trackDim[0],trackDim[1],5,5), dtype = object)
    for i in range(trackDim[0]):
        for j in range(trackDim[1]):
            for k in range(5):
                for l in range(5):
                    actionIndex = np.argmax(Qval[i,j,k,l])
                    Policy[i,j,k,l] = env.action_space[actionIndex]
    
    for ep in trange(int(NumEpisodes)):
        #B soft policy
        BPolicy = np.zeros((trackDim[0],trackDim[1],5,5), dtype = object)
        for i in range(trackDim[0]):
            for j in range(trackDim[1]):
                for k in range(5):
                    for l in range(5):
                        BPolicy[i][j][k][l] = random.choice(env.action_space)
        state,reward,done = env.reset()
        episodeDeets = []
        while True:
            bestAction = BPolicy[state[0],state[1],state[2],state[3]]
            action = MCPolicySelectorGreedy(Epsilon, bestAction)
            next_state, reward, done = env.step(state, action)
            episodeDeets.append((state,action,reward))
            if done == True:
                break
            state = next_state
        #initiate G and W
        G = 0
        W = 1
        #visited state tracker
        visited = []
        for episode in episodeDeets:
            G = (Gamma * G) + episode[2]
            CS = episode[0]
            CA = episode[1]
            CAindex = env.action_space.index(CA)
            Cval[CS[0],CS[1],CS[2],CS[3],CAindex] += W
            Qval[CS[0],CS[1],CS[2],CS[3],CAindex] += ((W / Cval[CS[0],CS[1],CS[2],CS[3],CAindex]) * (G - Qval[CS[0],CS[1],CS[2],CS[3],CAindex]))
            Policy[CS[0],CS[1],CS[2],CS[3]] = env.action_space[np.argmax(Qval[i,j,k,l])]
            if CA != Policy[CS[0],CS[1],CS[2],CS[3]]:
                continue 
            if CA != BPolicy[CS[0],CS[1],CS[2],CS[3]]:
                prob = Epsilon / 9
            else:
                prob = (1 - Epsilon) + (Epsilon / 9)
            W = W * (1 / prob)


# Multi_trials

def OffpolicyTrial(version, numTrials, numEpisodes, Epsilon, Gamma):
    G0total = []
    for trial in trange(numTrials, desc="Trial Progress", leave=False):
        G0 = FVMCRC(version, numEpisodes, Epsilon, Gamma)
        G0total.append(G0)
    print(G0total)
    return G0total


if __name__ == "__main__":

  G1 = OffpolicyTrial("v1", 10, 1000, 0.1, 0.9)
  G0 = MCRCpolicy("v1", 10, 1000, 0.1, 0.9)

  plot_performance([0.1], ["yellow"], np.array([G0]))
  plot_performance([0.1], ["yellow"], np.array([G1]))