import numpy as np
import random

# 그리드 월드의 크기 설정
grid_size = 5
n_actions = 6  # 가능한 액션의 수: 동, 서, 남, 북, 배터리 교체, 데이터 전송
n_states = grid_size * grid_size  # 상태의 수: 그리드 크기의 제곱

# 데이터 존재유무, 데이터 생성 된 카운트 저장
field = [[0, 0] for _ in range(grid_size * grid_size)]

# 각 상태들의 데이터 발생하지 않을 확률
percents = [1.0, 1.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 1.0, 1.0]

class UAV:
    battery = 25
    battery_threshold = 10  # 배터리 임계값 설정

battery_location = [[0, 0], [0, 4], [4, 0], [4, 4]]  # 배터리 충전소

n_episodes = 500  # 실행할 에피소드의 수

time_out = 10

results = []

# Q 테이블 초기화
Q = np.zeros((n_states, n_actions))  # 각 상태와 액션에 대해 0으로 초기화된 Q-테이블 생성

# 입실론 그리디 파라미터 설정
epsilon = 0.99  # 입실론 초기값, 무작위 액션을 선택할 확률
minus = 0.0001  # 각 에피소드 후에 입실론을 감소시킬 값
learning_rate = 0.2  # 학습률
discount_factor = 0.9  # 할인 계수

# 상태를 그리드 위치로 변환하는 함수, 1차원을 2차원으로
def state_to_position(state):
    return state // grid_size, state % grid_size

# 그리드 위치를 상태로 변환하는 함수, 2차원을 1차원으로
def position_to_state(position):
    return position[0] * grid_size + position[1]

# 액션에 따라 다음 상태를 결정하는 함수
def step(state, action):
    x, y = state_to_position(state)

    # 벽에 막혀있지 않을 때만 이동
    if action == 0 and y < grid_size - 1:  # 동쪽 이동
        y += 1
    elif action == 1 and y > 0:  # 서쪽 이동
        y -= 1
    elif action == 2 and x < grid_size - 1:  # 남쪽 이동
        x += 1
    elif action == 3 and x > 0:  # 북쪽 이동
        x -= 1

    UAV.battery -= 1  # 배터리 소모

    next_state = position_to_state((x, y))
    return_reward = -15 + g(next_state)  # 기본 보상값과 데이터가 오랜 시간 생성되어 있는 경우

    if action == 4 or action == 5:  # 액션이 배터리 충전 혹은 데이터 전송일 경우
        return_reward = def_action(next_state, action)

    return next_state, return_reward

def def_action(state, action):
    x, y = state_to_position(state)

    if action == 4:  # 배터리 교체
        if [x, y] in battery_location:  # 배터리 충전소에서 교체를 시도할 경우
            UAV.battery = 25
            return -1
        else:  # 배터리 충전소가 아닌 장소에서 교체를 시도할 경우
            return -15
    elif action == 5:  # 데이터 전송
        if percents[state] != 1.0:  # 데이터가 발생하지 않는 지점에서 데이터 전송 시도시
            return -15  # 보상 지급
        else:
            return try_send(state)  # 데이터가 발생하는 지점에서 데이터 전송 시도할 경우

def try_send(next_state):  # 데이터 전송 시도
    global check

    if field[next_state][0] == 1 :  # 데이터가 있다면
        field[next_state][0] = 0  # 데이터 제거
        field[next_state][1] = 0  # 카운트 초기화
        check += 1  # 생성된 데이터 갯수 체크

        return 30 - field[next_state][1]  # 데이터 전송 보상

    return -15  # 데이터가 없을 경우 보상

def g(next_state):
    global check_

    check_timeout = 0

    for i in range(len(field)) :
        if i == next_state :
            continue

        if field[i][1] > time_out :
            field[i][0] = 0  # 데이터 제거
            field[i][1] = 0  # 카운트 초기화

            check_timeout -= 5  # 타임아웃 패널티
            check_ += 1

    return check_timeout

# Q-러닝 알고리즘 실행
for episode in range(n_episodes):
    state = random.randint(0, n_states - 1)  # 무작위 초기 상태
    check = 0
    check_ = 1

    for _ in range(100):  # 각 에피소드에 대한 최대 스텝 수
        if random.random() < epsilon:
            action = random.choice([a for a in range(n_actions)])  # 모든 액션 중 무작위 선택
        else:
            action = np.argmax(Q[state, :])  # Q-테이블에서 최대값을 가진 액션 선택

        epsilon -= minus  # 입실론 값 감소

        next_state, reward = step(state, action)  # 선택된 액션으로 환경에서 한 스텝 진행

        # Q-러닝 업데이트
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state  # 상태 업데이트

    result = (check / check_) * 100
    results.append(result)

# 결과 출력
print(results)
