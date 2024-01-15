import numpy as np
import random

class GridWorld:
    def __init__(self, target_coordinates):
        self.grid_size = 10
        self.states = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.end_states = set(target_coordinates)  # 매개변수로 받은 좌표를 목표 지점으로 설정
        self.action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.actions = list(self.action_indices.keys())
        self.state = (0, 0)
        self.battery = 100

    def reset(self):
        self.state = (0, 0)
        self.battery = 100
        self.end_states = set(target_coordinates)  # 목표 지점 재설정
        return self.state

    def step(self, action_index):
        if self.battery <= 0:
            return self.state, -1, True  # 배터리 소진으로 에피소드 종료

        action = self.actions[action_index]
        next_state = self._get_next_state(self.state, action)

        if next_state in self.end_states:
            self.end_states.remove(next_state)
            reward = 10
        else:
            reward = -1

        self.state = next_state
        self.battery -= 1
        done = self.battery <= 0  # 배터리 소진 시에만 에피소드 종료
        return next_state, reward, done

    def _get_next_state(self, state, action):
        i, j = state
        if action == 'up':
            i = max(i - 1, 0)
        elif action == 'down':
            i = min(i + 1, self.grid_size - 1)
        elif action == 'left':
            j = max(j - 1, 0)
        elif action == 'right':
            j = min(j + 1, self.grid_size - 1)
        return (i, j)

    def render(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for end_state in self.end_states:
            grid[end_state[0]][end_state[1]] = 'G'
        grid[self.state[0]][self.state[1]] = 'A'

        print('Grid World:')
        for row in grid:
            print(' '.join(row))
        print()

def choose_action(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2, 3])
    else:
        return np.argmax(q_table[state])

def update_q_table(q_table, state, action_index, reward, next_state, alpha, gamma):
    predict = q_table[state][action_index]
    target = reward + gamma * np.max(q_table[next_state])
    q_table[state][action_index] += alpha * (target - predict)

def train_agent(episodes, alpha, gamma, epsilon, target_coordinates):
    env = GridWorld(target_coordinates)
    q_table = {state: [0 for _ in range(4)] for state in env.states}

    for episode in range(episodes):
        state = env.reset()
        done = False
        step_number = 0

        print(f"에피소드 {episode + 1} 시작")
        env.render()

        while not done:
            action_index = choose_action(state, q_table, epsilon)
            next_state, reward, done = env.step(action_index)
            update_q_table(q_table, state, action_index, reward, next_state, alpha, gamma)
            
            state = next_state
            step_number += 1


            if episode >= 900 :
                print(f"스텝 {step_number}")
                env.render()
                input()

            if done:
                print(f"에피소드 {episode + 1} 종료\n")

    return q_table

# 하이퍼파라미터 설정
episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# 목표 지점 좌표 설정
target_coordinates = [(1, 1), (3, 4), (6, 8)]

# 에이전트 훈련
q_table = train_agent(episodes, alpha, gamma, epsilon, target_coordinates)
