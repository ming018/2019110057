import numpy as np
import random

class GridWorld:
    def __init__(self):
        self.grid_size = 3
        self.goal_states = [(0, 0), (2, 2)]
        self.state = None
        self.reset()

    def reset(self):
        self.state = (1, 1)  # 중간 위치에서 시작
        return self.state_to_index(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0: y = max(y - 1, 0)  # 상
        elif action == 1: y = min(y + 1, self.grid_size - 1)  # 하
        elif action == 2: x = max(x - 1, 0)  # 좌
        elif action == 3: x = min(x + 1, self.grid_size - 1)  # 우

        self.state = (x, y)
        done = self.state in self.goal_states
        reward = 1 if done else -1
        return self.state_to_index(self.state), reward, done

    def state_to_index(self, state):
        return state[0] + self.grid_size * state[1]

def choose_action(state_index, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2, 3])  # 무작위 행동 선택
    else:
        return np.argmax(q_table[state_index])  # Q-테이블에서 최적의 행동 선택

def update_q_table(q_table, state_index, action, reward, next_state_index, alpha, gamma):
    predict = q_table[state_index][action]
    target = reward + gamma * np.max(q_table[next_state_index])
    q_table[state_index][action] += alpha * (target - predict)

# 환경 및 Q-테이블 초기화
env = GridWorld()
q_table = np.zeros((env.grid_size ** 2, 4))

# 학습 파라미터 설정
alpha = 0.1  # 학습률
gamma = 0.9  # 할인 계수
epsilon = 0.1  # 탐색 확률
episodes = 1000  # 총 에피소드 수

# Q-러닝 학습 과정
for episode in range(episodes):
    state_index = env.reset()
    done = False

    while not done:
        action = choose_action(state_index, q_table, epsilon)
        next_state_index, reward, done = env.step(action)
        
        update_q_table(q_table, state_index, action, reward, next_state_index, alpha, gamma)
        state_index = next_state_index

# 학습된 Q-테이블 출력
print("Q-Table:")
print(q_table)
