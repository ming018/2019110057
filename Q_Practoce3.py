# 이거는 R매트릭스가 가변하면서 입실론 그리디 정책을 사용한 예제

import numpy as np
import random

# 환경 설정
num_states = 5
num_actions = 5

# R-matrix 및 Q-테이블 초기화
R = np.random.randint(-1, 100, (num_states, num_actions))
Q = np.zeros((num_states, num_actions))

# 학습 매개변수
gamma = 0.8  # 할인 계수
learning_rate = 0.1  # 학습률
epsilon = 0.1  # 탐험율
num_episodes = 1000  # 총 에피소드 수

# 행동 선택 함수 (입실론-그리디)
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)  # 무작위 행동
    else:
        return np.argmax(Q[state, :])  # 최적의 행동

# 환경에서 R-matrix 업데이트 함수
def update_R_matrix():
    R[random.randint(0, num_states-1), random.randint(0, num_actions-1)] = random.randint(-1, 100)

# Q-learning 학습 과정
for episode in range(num_episodes):
    state = random.randint(0, num_states-1)  # 초기 상태 선택

    for _ in range(100):
        action = choose_action(state, epsilon)  # 행동 선택
        next_state = random.randint(0, num_states-1)  # 무작위 다음 상태 선택
        reward = R[state, action]  # 현재 상태와 행동에 대한 보상

        # Q-테이블 업데이트
        Q[state, action] += learning_rate * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state  # 상태 업데이트

    # R-matrix 업데이트
    update_R_matrix()

# 최종 Q-테이블 출력
print("Final Q-Table:")
print(Q)
