import numpy as np
import random

# 환경 설정
grid_size = 10
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
actions = ['up', 'down', 'left', 'right']
goal_locations = {'A': (1, 2), 'B': (4, 5), 'C': (7, 8), 'D': (3, 9)}
charging_stations = [(0, 0), (9, 9), (9, 0), (0, 9), (4, 4)]  # 충전소 위치

# Q-테이블 초기화
Q = {}
for state in states:
    for next_goal in goal_locations:
        for action in actions:
            Q[(state, next_goal, action)] = 0

# 배터리 관련 설정
max_battery = 50 # 최대 배터리 수준
battery_drain = 1  # 행동당 배터리 소모량
charging_amount = 15  # 충전소에서 충전되는 양

# 맨해튼 거리를 계산하는 함수
def calculate_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

# 가장 가까운 목표를 찾는 함수 (수정됨)
def find_nearest_goal(state, remaining_goals, visited_goals):
    nearest_goal = None
    min_distance = float('inf')
    for goal in remaining_goals:
        if goal not in visited_goals:  # 아직 방문하지 않은 목표에 우선순위 부여
            distance = calculate_distance(state, goal_locations[goal])
            if distance < min_distance:
                min_distance = distance
                nearest_goal = goal
    return nearest_goal if nearest_goal is not None else random.choice(list(remaining_goals))

# 그리디 월드 출력 함수
def print_grid(state, goal_locations, charging_stations, battery_level):
    grid = [['-' for _ in range(grid_size)] for _ in range(grid_size)]
    for goal, loc in goal_locations.items():
        x, y = loc
        grid[x][y] = goal
    for station in charging_stations:
        x, y = station
        grid[x][y] = 'C'  # 충전소 위치
    x, y = state
    grid[x][y] = 'E'  # 에이전트의 현재 위치
    for row in grid:
        print(' '.join(row))
    print(f'남은 배터리: {battery_level}\n')

# 주어진 행동에 따라 상태를 업데이트하는 함수
def update_state(state, action):
    x, y = state
    if action == 'up':
        x = max(0, x-1)
    elif action == 'down':
        x = min(grid_size-1, x+1)
    elif action == 'left':
        y = max(0, y-1)
    elif action == 'right':
        y = min(grid_size-1, y+1)
    return (x, y)

# 배터리 업데이트 및 보상 함수
def update_battery_and_reward(state, next_state, battery_level):
    reward = 0
    if next_state in charging_stations:
        reward = 2  # 충전소에서 충전할 때 추가 보상
    if next_state in goal_locations.values():
        reward = 1  # 목표 지점에 도달했을 때의 보상

    if state in charging_stations:
        battery_level = min(max_battery, battery_level + charging_amount)  # 충전
    else:
        battery_level = max(0, battery_level - battery_drain)  # 배터리 소모

    return battery_level, reward

# Q-러닝 업데이트 로직
def update_q_table(Q, state, next_goal, action, reward, next_state, alpha, gamma):
    old_value = Q[(state, next_goal, action)]
    next_max = max([Q[(next_state, next_goal, a)] for a in actions])

    new_value = old_value + alpha * (reward + gamma * next_max - old_value)
    Q[(state, next_goal, action)] = new_value

# 에피소드 실행 함수 (수정됨)
def run_episode(Q, alpha, gamma, epsilon):
    state = random.choice(states)  # 무작위로 시작 상태 선택
    battery_level = max_battery  # 배터리 수준 초기화
    step = 0  # 이동 횟수 초기화
    remaining_goals = set(goal_locations.keys())  # 남은 목표 지점들
    visited_goals = set()  # 방문한 목표 지점들

    while remaining_goals and battery_level > 0:
        current_goal = find_nearest_goal(state, remaining_goals, visited_goals)

        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            q_values = [Q[(state, current_goal, a)] for a in actions]
            max_q = max(q_values)
            action = random.choice([actions[i] for i, q in enumerate(q_values) if q == max_q])

        next_state = update_state(state, action)
        battery_level, reward = update_battery_and_reward(state, next_state, battery_level)

        # Q-값 업데이트
        update_q_table(Q, state, current_goal, action, reward, next_state, alpha, gamma)

        if next_state in goal_locations.values():
            visited_goal = [goal for goal, loc in goal_locations.items() if loc == next_state][0]
            visited_goals.add(visited_goal)  # 목표 지점 방문 시 추가
            if visited_goal in remaining_goals:
                remaining_goals.remove(visited_goal)  # 방문한 목표 지점 제거

        if battery_level == 0:
            break  # 배터리가 소진되면 에피소드 종료

        state = next_state
        step += 1

        # if episode >= 95 :
        #     print_grid(state, goal_locations, charging_stations, battery_level)
        #     input()

        if step >= 500 :
            break

    return step, len(remaining_goals) == 0  # 이동 횟수 반환 및 목표 달성 여부

# 학습 파라미터
alpha = 0.1
gamma = 0.9
epsilon = 0.4

# 학습 과정
successful_episodes = 0  # 성공적으로 완료된 에피소드 수

success_episode = []

for episode in range(100):
    steps, success = run_episode(Q, alpha, gamma, epsilon)
    if success:
        successful_episodes += 1  # 성공적으로 목표를 달성한 경우 카운트 증가
        success_episode.append(episode)
    print(f"에피소드 {episode} 완료, 총 스텝 수: {steps}")

print(f"성공적으로 완료된 에피소드 수: {successful_episodes}")
print("성공한 에피소드 회차", success_episode)
