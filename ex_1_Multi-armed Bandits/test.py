import numpy as np

# Define the grid size based on the image provided
grid_size = (11, 11)  # 11x11 grid
goal_state = (10, 10)  # Goal state as per the image
start_state = (0, 0)  # Start state as per the image

# Define walls based on the image provided
# A wall is represented by a list of coordinates where the agent cannot go
walls = [
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
    (9, 2), (10, 2), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
    (7, 5), (8, 5), (9, 5), (10, 5), (3, 7), (4, 7), (5, 7),
    (6, 7), (7, 7), (8, 7), (9, 7), (4, 9), (5, 9), (6, 9),
    (7, 9), (8, 9), (9, 9), (10, 9)
]

# Define the state space S and the action space A
S = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1]) if (i, j) not in walls]
A = ['up', 'down', 'left', 'right']

# Initialize the dynamics function P as a dictionary
P = {}

# Helper function to determine if the next state is valid (not a wall or outside the grid)
def is_valid_state(s):
    return s not in walls and (0 <= s[0] < grid_size[0]) and (0 <= s[1] < grid_size[1])

# Helper function to determine the next state given action a
def next_state(s, a):
    potential_next_state = {
        'up': (s[0] - 1, s[1]),
        'down': (s[0] + 1, s[1]),
        'left': (s[0], s[1] - 1),
        'right': (s[0], s[1] + 1),
    }.get(a, s)

    return potential_next_state if is_valid_state(potential_next_state) else s

# Generate the dynamics function table
for s in S:
    for a in A:
        if s == goal_state:
            # Teleport to start state with a reward of +1
            P[(s, a)] = [(start_state, 1)]
        else:
            s_prime = next_state(s, a)
            reward = -1 if s_prime == s else 0  # Penalty for hitting a wall or -1
            P[(s, a)] = [(s_prime, reward)]

# For brevity, let's print the transitions for the state (1,1) as an example
# Note: (1,1) is adjacent to a wall to the left and up, based on the image provided
example_transitions = {a: P[((1, 1), a)] for a in A}
example_transitions
# Generate the dynamics function table for all valid states
full_dynamics = {}
for s in S:
    for a in A:
        if s == goal_state:
            # Teleport to start state with a reward of +1
            full_dynamics[(s, a)] = [(start_state, 1)]
        else:
            s_prime = next_state(s, a)
            reward = -1 if s_prime == s else 0  # Penalty for hitting a wall or -1
            full_dynamics[(s, a)] = [(s_prime, reward)]

# Function to print the dynamics for a given state
def print_dynamics_for_state(state):
    for a in A:
        transitions = full_dynamics.get((state, a), [])
        if transitions:
            print(f"From state {state} taking action '{a}':")
            for (s_prime, reward) in transitions:
                print(f"  -> To state {s_prime} with reward {reward}")

# We can now print the dynamics for a few example states
# Let's print the dynamics for the states (0,0), (1,1), and the goal state (10,10)
print_dynamics_for_state((0, 0))
print_dynamics_for_state((1, 1))
print_dynamics_for_state(goal_state)
