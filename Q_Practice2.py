# R값이 변경 되는 matrix를 포함한 Q러닝
import numpy as np
import random

# 환경 설정
num_states = 5
num_actions = 4

# R-matrix 및 Q-테이블 초기화
R = np.random.randint(-5, 0, (num_states, num_actions))
Q = np.zeros((num_states, num_actions))
world = [0 for _ in range(num_states * num_states)]
world[len(world) // 2] = 1

# 학습 매개변수
gamma = 0.8  # 할인 계수
learning_rate = 0.1  # 학습률
num_episodes = 1000  # 총 에피소드 수

# 환경에서 R-matrix 업데이트 함수
def update_R_matrix():
    # R-matrix를 업데이트하는 로직을 여기에 구현
    # 예: 무작위로 R-matrix의 일부 값을 변경
    
    for i in range (num_states) :
        R[i, random.randint(0, 4)] = 1

def move(location, a) :
    # P는 1로 설정
    # 4방향으로 알파버전하고 추후 여유롭다 싶으면 8방향으로 수정


    # 동 서 남 북 : 0 1 2 3
    if a == 0 : # 동
        if location % num_states == 4 :
            pass
        location[location] = 0

        location += 1

        world[location] = 1
       
    
    elif a == 1 : # 서
        if location % num_states == 0 :
            pass
        
        location[location] = 0

        location -= 1

        world[location] = 1
        
    
    elif a == 2 : # 남
        if location >= 20 :
            pass
        
        location[location] = 0

        location += num_states

        world[location] = 1

    elif a == 3 : # 북
        if location >= 4 :
            pass
        location[location] = 0

        location -= num_states

        world[location] = 1

    return location

# Q-learning 학습 과정
for episode in range(num_episodes):
    state = random.randint(0, num_states-1)  # 초기 상태 선택

    for _ in range(100):
        action = random.randint(0, num_actions-1)  # 무작위 행동 선택
        next_state = random.randint(0, num_states-1)  # 무작위 다음 상태 선택
        reward = R[state, action]  # 현재 상태와 행동에 대한 보상

        # Q-테이블 업데이트
        Q[state, action] += learning_rate * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state  # 상태 업데이트
    
    print('---------------------')
    print(R)
    print()
    print(Q)
    print('---------------------')
    
    input()

    # R-matrix 업데이트
    # update_R_matrix()