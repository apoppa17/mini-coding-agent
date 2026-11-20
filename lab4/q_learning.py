import numpy as np

# Project Environment
grid_size = 4
start_position = (0, 0)
obstacle_position = (1, 1)
negative_reward_position = (3, 1)
goal_position = (3, 3)

# Q-learning parameters
living_reward = -0.1
discount_factor = 0.95
learning_rate = 0.1
exploration_rate = 0.1

# Initialize Q-table
q_table = np.zeros((grid_size, grid_size, 4))

# Actions: up, down, left, right
actions = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

def get_next_state(state, action):
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    if next_state[0] < 0 or next_state[0] >= grid_size or next_state[1] < 0 or next_state[1] >= grid_size or next_state == obstacle_position:
        return state
    return next_state

def get_reward(state):
    if state == negative_reward_position:
        return -1
    elif state == goal_position:
        return 10
    else:
        return living_reward

def choose_action(state):
    if np.random.rand() < exploration_rate:
        return np.random.randint(4)
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, next_state, reward):
    q_table[state][action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action])

# Q-learning algorithm
num_episodes = 10000
for episode in range(num_episodes):
    state = start_position
    for _ in range(100):  # Maximum episode length
        action = choose_action(state)
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        update_q_table(state, action, next_state, reward)
        state = next_state
        if state == goal_position:
            break

# Print the learned Q-table
print(q_table)
