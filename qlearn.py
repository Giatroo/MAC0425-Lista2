import numpy as np

r = np.asarray([[-1, -1, -1, 100], [-1, -100, -1, -1], [-1, -1, -1, 10]])

#  q_value = np.random.randn(r.shape[0], r.shape[1], 4)  # 3 linhas, 4 colunas, 4 ações possíveis
q_value = np.zeros((r.shape[0], r.shape[1], 4))  # 3 linhas, 4 colunas, 4 ações possíveis

actions = {
    0: np.asarray([-1, 0]),  # subir
    1: np.asarray([1, 0]),  # descer
    2: np.asarray([0, -1]),  # esquerda
    3: np.asarray([0, 1]),  # direita
}
actions_str = {
    0: "^",
    1: "v",
    2: "<",
    3: ">",
}

learning_rate = 1
discount = 0.90
#  discount = 1


def update_q_values(q_value):
    for row in range(q_value.shape[0]):
        for column in range(q_value.shape[1]):
            for action in range(q_value.shape[2]):
                if (row, column) == (0, 3):
                    q_value[row][column][action] = r[row][column]
                    continue

                cur_state = np.asarray([row, column])
                cur_action = actions[action]
                new_state = cur_state + cur_action

                if (
                    new_state[0] < 0
                    or new_state[0] > 2
                    or new_state[1] < 0
                    or new_state[1] > 3
                ):
                    new_state = np.asarray([row, column])

                #  print(f'From state {cur_state} taking action {cur_action} we achieve {new_state}.')
                #  print(f'The q-values are: {q_value[new_state[0]][new_state[1]]}')
                max_action_q = max(q_value[new_state[0]][new_state[1]])
                #  print(f'max-q-value is {max_action_q}')
                #  print('-'*20)

                cur_q = q_value[row][column][action]
                reward = r[row][column]

                new_q = (1 - learning_rate) * cur_q + learning_rate * (
                    reward + discount * max_action_q
                )
                q_value[row][column][action] = new_q
    return q_value


def get_policy(q_value):
    policy = np.empty_like(r).astype(str)

    for row in range(r.shape[0]):
        for column in range(r.shape[1]):
            policy[row][column] = actions_str[np.argmax(q_value[row][column])]
    return policy


def itera(q_value, debug=True):
    if debug:
        print(f"q-values atuais:\n {q_value}")
        print("-" * 40)
        print(f"política atual:\n {get_policy(q_value)}")
        print("-" * 40)
    return update_q_values(q_value)


stop = 'n'
while stop != 'y':
    itera(q_value)
    stop = input('Deseja parar? (y/N)')

#  print(q_value[0][2])
