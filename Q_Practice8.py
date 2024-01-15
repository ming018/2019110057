import numpy as np
import random

class GridWorld:
    def __init__(self, size, data_points, charging_stations):
        self.size = size
        self.agent_position = (0, 0)
        self.data_points = set(data_points)  # 데이터 수집 지점
        self.charging_stations = set(charging_stations)  # 충전소 위치

    def find_nearest_data_point(self, position):
        """
        현재 위치에서 가장 가까운 미방문 데이터 발생지를 찾습니다.
        """
        unvisited = [p for p in self.data_points if p not in self.visited_data_points]
        if not unvisited:
            return None

        nearest_point = min(unvisited, key=lambda p: self.distance(position, p))
        return nearest_point

    def distance(pos1, pos2):
        """
        두 위치 사이의 거리를 계산합니다.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])    

    def reset(self):
        """
        환경을 초기 상태로 리셋합니다.
        :return: 초기화된 에이전트의 위치를 반환합니다.
        """
        self.agent_position = (0, 0)
        self.visited_data_points = set()  # 방문한 데이터 장소를 추적하기 위한 집합
        return self.agent_position
    
    def check_visited_data_points(self, position):
        """
        에이전트가 데이터 장소에 도착했는지 확인합니다.
        """
        if position in self.data_points:
            self.visited_data_points.add(position)

    def get_valid_actions(self):
        """
        에이전트의 현재 위치를 기반으로 가능한 행동들을 계산합니다.
        :return: 현재 위치에서 수행 가능한 행동들의 리스트를 반환합니다.
        """
        valid_actions = []
        x, y = self.agent_position

        if y < self.size - 1: valid_actions.append(0)  # 동쪽 이동 가능
        if y > 0: valid_actions.append(1)  # 서쪽 이동 가능
        if x < self.size - 1: valid_actions.append(2)  # 남쪽 이동 가능
        if x > 0: valid_actions.append(3)  # 북쪽 이동 가능

        return valid_actions

    def step(self, action):
        """
        에이전트가 선택한 행동을 수행하고 결과를 반환합니다.
        :param action: 에이전트가 수행할 행동입니다.
        :return: 새로운 에이전트의 위치를 반환합니다.
        """
        x, y = self.agent_position

        if action == 0: y = min(self.size - 1, y + 1)  # 동쪽으로 이동
        elif action == 1: y = max(0, y - 1)  # 서쪽으로 이동
        elif action == 2: x = min(self.size - 1, x + 1)  # 남쪽으로 이동
        elif action == 3: x = max(0, x - 1)  # 북쪽으로 이동

        self.agent_position = (x, y)
        return self.agent_position

    def render(self, battery_level, total_reward):
        """
        현재 그리드의 상태와 에이전트의 정보를 시각적으로 출력합니다.
        :param battery_level: 현재 에이전트의 배터리 수준입니다.
        :param total_reward: 에이전트가 현재까지 얻은 총 보상입니다.
        """ 
        grid = [['-' for _ in range(self.size)] for _ in range(self.size)]
        for x, y in self.data_points:
            grid[x][y] = 'D'  # 데이터 지점
        for x, y in self.charging_stations:
            grid[x][y] = 'C'  # 충전소

        x, y = self.agent_position
        grid[x][y] = 'A'  # 에이전트 위치

        print("\n".join([" ".join(row) for row in grid]))
        print(f"Battery: {battery_level}, Total Reward: {total_reward}")
        print()

class QLearningAgent:
    def __init__(self, num_states, num_actions, epsilon, alpha, gamma, minus):
        self.Q_table_data_collection = np.zeros((num_states, num_actions))
        self.Q_table_battery_charging = np.zeros((num_states, num_actions))
        self.epsilon_data = epsilon
        self.epsilon_battery = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.battery_level = 100
        self.battery_threshold = 20
        self.minus = minus

    def choose_action(self, state, valid_actions):
        current_position = (state // 10, state % 10)
        if self.battery_level >= self.battery_threshold:
            q_table = self.Q_table_data_collection
            temp_epsilon = self.epsilon_data
            nearest_data_point = env.find_nearest_data_point(current_position)
            if nearest_data_point is not None:
                chosen_action = self.determine_action_towards_point(current_position, nearest_data_point, valid_actions)
                return chosen_action
        else:
            q_table = self.Q_table_battery_charging
            temp_epsilon = self.epsilon_battery

        if random.uniform(0, 1) < temp_epsilon:
            return random.choice(valid_actions)
        else:
            q_values = q_table[state, valid_actions]
            max_q_index = np.argmax(q_values)
            return valid_actions[max_q_index]

    def determine_action_towards_point(self, current_position, target_position, valid_actions):
        x_current, y_current = current_position
        x_target, y_target = target_position

        # x축과 y축 방향으로의 차이 계산
        delta_x = x_target - x_current
        delta_y = y_target - y_current

        # x축과 y축 방향으로 이동해야 하는지 결정
        if abs(delta_x) > abs(delta_y):
            # x축 방향의 이동이 우선시 됨
            if delta_x > 0 and 2 in valid_actions:  # 남쪽으로 이동
                return 2
            elif delta_x < 0 and 3 in valid_actions:  # 북쪽으로 이동
                return 3
        else:
            # y축 방향의 이동이 우선시 됨
            if delta_y > 0 and 0 in valid_actions:  # 동쪽으로 이동
                return 0
            elif delta_y < 0 and 1 in valid_actions:  # 서쪽으로 이동
                return 1

        # 만약 직접적인 경로로 이동할 수 없는 경우, 가능한 행동 중 하나를 무작위로 선택
        return random.choice(valid_actions)

    def learn(self, state, action, reward, next_state, next_valid_actions):
        if self.battery_level >= self.battery_threshold:
            q_table = self.Q_table_data_collection
        else:
            q_table = self.Q_table_battery_charging

        next_max = np.max(q_table[next_state, next_valid_actions])
        q_table[state, action] += self.alpha * (reward + self.gamma * next_max - q_table[state, action])

    def update_battery(self):
        self.battery_level -= 1

    def print_q_tables(self, choice):
        np.set_printoptions(precision=2, suppress=True)
        if choice == 1:
            print("Q-table for Data Collection with Optimal Actions:")
            for state in range(len(self.Q_table_data_collection)):
                optimal_action = np.argmax(self.Q_table_data_collection[state])
                print(f"State {state}, Optimal Action: {optimal_action}")
        elif choice == 2:
            print("\nQ-table for Battery Charging with Optimal Actions:")
            for state in range(len(self.Q_table_battery_charging)):
                optimal_action = np.argmax(self.Q_table_battery_charging[state])
                print(f"State {state}, Optimal Action: {optimal_action}")
                


 # 환경 및 에이전트 초기화
data_points = [(2, 3), (4, 4), (3, 7)]
charging_stations = [(0, 9), (9, 0), (9, 9)]
env = GridWorld(10, data_points, charging_stations)
agent = QLearningAgent(num_states=100, num_actions=4, epsilon=0.9, alpha=0.5, gamma=0.9, minus=0.001)

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    done = False
    agent.battery_level = 100

    while not done:
        valid_actions = env.get_valid_actions()
        current_state_index = state[0] * 10 + state[1]
        action = agent.choose_action(current_state_index, valid_actions)
        next_state = env.step(action)
        next_state_index = next_state[0] * 10 + next_state[1]

        # 보상 로직
        reward = 0
        if agent.battery_level >= agent.battery_threshold:
            if next_state in env.data_points and next_state not in env.visited_data_points:
                reward = 10  # 새로운 데이터 지점 방문
                env.visited_data_points.add(next_state)  # 방문한 데이터 지점 추가
        else:
            if next_state in env.charging_stations:
                reward = 20  # 충전소 도달
                agent.battery_level = 100  # 배터리 재충전

        total_reward += reward
        agent.learn(current_state_index, action, reward, next_state_index, env.get_valid_actions())
        state = next_state

        if agent.battery_level <= 0:
            done = True

        agent.update_battery()

        # 상태 출력 (옵션)
        if episode >= 95:
            env.render(agent.battery_level, total_reward)
            print(reward)
            num = input()
            if num == "1" or num == "2":
                agent.print_q_tables(int(num))

    print(f'\n==============================================\n에피소드 {episode} 종료')               

# # 환경 및 에이전트 초기화
# data_points = [(2, 3), (4, 4), (3, 7)]
# charging_stations = [(0, 9), (9, 0), (9, 9)]
# env = GridWorld(10, data_points, charging_stations)
# agent = QLearningAgent(num_states=100, num_actions=4, epsilon=0.9, alpha=0.5, gamma=0.9, minus=0.001)

# for episode in range(1000):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     agent.battery_level = 100

#     for _ in range(100) :
#     #while True :
#         valid_actions = env.get_valid_actions()
#         current_state_index = state[0] * 10 + state[1]
#         action = agent.choose_action(current_state_index, valid_actions)
#         next_state = env.step(action)
#         next_state_index = next_state[0] * 10 + next_state[1]

#         # 에이전트가 데이터 지점을 발견하면, 이를 기록
#         env.check_visited_data_points(state)

#         # 보상 로직
#         reward = 0
#         if agent.battery_level >= agent.battery_threshold:
#             if next_state in env.data_points and next_state not in env.visited_data_points:
#                 reward = -1  # 새로운 데이터 지점 방문
#                 env.visited_data_points.add(next_state)  # 방문한 데이터 지점 추가
#             else :
#                 if next_state not in env.data_points :
#                     reward = -5
#         else:
#             if next_state in env.charging_stations:
#                 reward = -5  # 충전소 도달
#                 agent.battery_level = 100  # 배터리 재충전
#             else :
#                 reward = -10
#         if agent.battery_level <= 0 :
#             reward = -50
            
        

#         total_reward += reward
#         agent.learn(current_state_index, action, reward, next_state_index, env.get_valid_actions())
#         state = next_state

#         if episode >= 95 :
#         #if True :
#             env.render(agent.battery_level, total_reward)
            
#             print(reward)
#             num = input()

#             if num == "1" or num == "2":
#                 agent.print_q_tables(int(num))

#         if agent.battery_level <= 0 :
#             break

#         agent.update_battery()

#     print(f'\n==============================================\n에피소드 {episode} 종료')