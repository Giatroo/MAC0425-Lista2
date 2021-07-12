import random as rng

import numpy as np

ex = input("Qual o exercício? (1/2) ")

initial_state = (2, 0)
final_states = [(0, 3)]
r = np.asarray([[-1, -1, -1, 100], [-1, -100, -1, -1], [-1, -1, -1, -1]])
if ex.strip() == "2":
    r[2][3] = 10

print(f"As recompensas são:\n{r}\n")

# q_values = np.random.randn(r.shape[0], r.shape[1], 4)
q_values = np.zeros((r.shape[0], r.shape[1], 4))


actions = {
    0: np.asarray([0, -1]),  # esquerda
    1: np.asarray([-1, 0]),  # subir
    2: np.asarray([0, 1]),  # direita
    3: np.asarray([1, 0]),  # descer
}
actions_str = {
    0: "<",
    1: "^",
    2: ">",
    3: "v",
}

ALPHA = 1
GAMMA = 0.9
EPS = 0.3  # chances of doing something different


def choose_action(q_values, cur_state):
    if rng.random() < EPS:
        return rng.randrange(len(actions))
    return np.argmax(q_values[cur_state])


def is_inside(loc, matrix):
    shape = np.asarray(matrix.shape)
    loc = np.asarray(loc)
    return all(np.zeros_like(shape) <= loc) and all(loc < shape)


def update_q_values(q_values, cur_state, action_taken, reward):
    # If the state is a final state, the q-value is the reward
    if cur_state in final_states:
        q_values[(*cur_state, action_taken)] = r[cur_state]
        return q_values

    new_state = cur_state + actions[action_taken]
    if not is_inside(new_state, r):
        new_state = cur_state
    new_state = tuple(new_state)

    # updating the q-value
    max_action_q = np.max(q_values[new_state])
    cur_q = q_values[(*cur_state, action_taken)]

    new_q = (1 - ALPHA) * cur_q + ALPHA * (reward + GAMMA * max_action_q)
    q_values[(*cur_state, action_taken)] = new_q
    return q_values


def get_policy_str(q_values):
    policy = np.empty_like(r).astype(str)

    for row in range(r.shape[0]):
        for column in range(r.shape[1]):
            policy[row][column] = actions_str[np.argmax(q_values[row][column])]
    return policy


def iteration(cur_state, q_values, debug=True):
    if debug:
        print(f"estado atual: {cur_state}")
        print("-" * 40)
        print(f"q-values atuais:\n {q_values}")
        print("-" * 40)
        print(f"política atual:\n {get_policy_str(q_values)}")
        print("-" * 40)

    # First, we choose an action based on the current q_values
    chosen_action = choose_action(q_values, cur_state)

    # after choosing the action, there are chances that the agent will not go to
    # the right direction (the one it chosen).
    rand_num = rng.random()
    if rand_num < 0.8:
        new_state = cur_state + actions[chosen_action]
    elif rand_num < 0.9:
        new_action = chosen_action if chosen_action > 0 else len(actions) - 1
        new_state = cur_state + actions[new_action]
    else:
        new_state = cur_state + actions[(chosen_action + 1) % len(actions)]

    # if the action leads to a state outside our grid, we stay in the same place
    if not is_inside(new_state, r):
        new_state = cur_state

    reward = r[tuple(cur_state)]

    return tuple(new_state), update_q_values(q_values, cur_state, chosen_action, reward)


stop = "n"
num_iter = 0

for final_state in final_states:
    q_values[final_state] = np.full((1, len(actions)), r[final_state])

print(q_values)
input()

while stop.strip() != "y":
    num_iter += 1
    cur_state = initial_state
    while cur_state not in final_states:
        cur_state, q_values = iteration(cur_state, q_values)
    stop = input("Deseja parar? (y/N) ")

#  print(np.max(q_value, axis=2))

print(f"\nForam {num_iter} iterações.")
