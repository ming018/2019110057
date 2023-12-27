import numpy as np

# 환경 설정: 그리드 크기, 행동 개수
grid_size = 5
num_actions = 4  # 동, 서, 남, 북

# Q-매트릭스 초기화
Q = np.zeros((grid_size * grid_size, num_actions))

# 학습 매개변수 설정
learning_rate = 0.1
discount_factor = 0.99
num_episodes = 1000

# 이동 함수 정의
def move(state, action):
    row, col = divmod(state, grid_size)
    if action == 0 and col < grid_size - 1:  # 동
        col += 1
    elif action == 1 and col > 0:  # 서
        col -= 1
    elif action == 2 and row < grid_size - 1:  # 남
        row += 1
    elif action == 3 and row > 0:  # 북
        row -= 1
    return row * grid_size + col

# 에이전트가 환경에서 학습하는 과정
for episode in range(num_episodes):
    # 초기 상태 설정
    state = np.random.randint(0, grid_size * grid_size)
    
    # 한 에피소드 내에서의 학습 과정
    while True:
        # 임의로 행동 선택
        action = np.random.randint(0, num_actions)

        # 새로운 상태와 보상 받기
        next_state = move(state, action)
        reward = np.random.rand()  # 임의의 보상 설정

        # Q-값 업데이트
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

        # 다음 상태로 이동
        state = next_state

        # 간단한 예제를 위해 무작위로 에피소드 종료
        if np.random.rand() < 0.1:
            break


# 업데이트된 Q-매트릭스 출력
print("Q-매트릭스:")
print(Q)