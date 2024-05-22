from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import random

def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    There are a couple of ways to create Gym environments of the different variants of Windy Grid World.
    1. Create separate classes for each env and register each env separately.
    2. Create one class that has flags for each variant and register each env separately.

        Example:
        (Original)     register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
        (King's moves) register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)

        The kwargs will be passed to the entry_point class.

    3. Create one class that has flags for each variant and register env once. You can then call gym.make using kwargs.

        Example:
        (Original)     gym.make("WindyGridWorld-v0")
        (King's moves) gym.make("WindyGridWorld-v0", **kwargs)

        The kwargs will be passed to the __init__() function.

    Choose whichever method you like.
    """
    # TODO
    register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv", max_episode_steps=10000)


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class Action_Kings(IntEnum):
    """Action"""
    
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    L_U = 4
    L_D = 5
    R_U = 6
    R_D = 7
    N = 8

def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action_Kings.LEFT: (-1, 0),
        Action_Kings.DOWN: (0, -1),
        Action_Kings.RIGHT: (1, 0),
        Action_Kings.UP: (0, 1),
        Action_Kings.L_U: (-1, 1),
        Action_Kings.L_D: (-1, -1),
        Action_Kings.R_D: (1, -1),
        Action_Kings.R_U: (1, 1),
        Action_Kings.N: (0, 0)
    }
    return mapping[action]


class WindyGridWorldEnv(Env):
    def __init__(self, types: int):
        """Windy grid world gym environment
        This is the template for Q4a. You can use this class or modify it to create the variants for parts c and d.
        """
        self.kings_move = False
        self.stochastic_wind = False

        if types == 1:
            self.kings_move = True

        if types == 2:
            self.kings_move = True
            self.stochastic_wind = True
        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        # TODO define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        W = {}

        for i in range(self.rows):
            for j in range(self.cols):
                if i in [0, 1, 2, 9]:
                    W[i,j] = 0
                elif i in [3, 4, 5, 8]:
                    W[i,j] = 1
                else:
                    W[i,j] = 2
        self.wind = W
        # print(self.wind)

        if self.kings_move:
            self.action_space = spaces.Discrete(len(Action_Kings))
        else:
            self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # TODO
        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 0.0
        else:
            done = False
            reward = -1.0
        # print(self.wind)
        A = actions_to_dxdy(action)

        
        Nx = self.agent_pos[0] + A[0]
        Ny = self.agent_pos[1] + A[1]
        next_pos = (Nx, Ny)
        # # print(next_pos)
        y = self.wind[self.agent_pos[0],self.agent_pos[1]]


        # next_pos = tuple(map(sum, zip(self.agent_pos, actions_to_dxdy(action))))

        if next_pos not in self.observation_space:
                next_pos = self.agent_pos

        if self.stochastic_wind and y>0:
                probs = [y-1, y, y+1]
                st_wind = random.choice(probs)
                next_pos_w = (next_pos[0], next_pos[1] + st_wind)
        else:
                next_pos_w = (next_pos[0], next_pos[1] + y)
    

        if next_pos_w not in self.observation_space:
                next_pos_w = next_pos  

        self.agent_pos = next_pos

        # self.agent_pos = next_pos
        # print(action)
        # print(A)
        # print(self.agent_pos)

        return self.agent_pos, reward, done, {}
