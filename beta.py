import numpy as np
import random

# 환경 설정
num_states = 6
num_actions = 4

# R-matrix 및 Q-테이블 초기화
R = np.zeros((num_states * num_states, num_actions))
Q = np.zeros((num_states * num_states, num_actions))

# 센싱 데이터의 유무와 카운트를 담는 배열
field = [[0, 0] for _ in range(num_states * num_states)]
check_date = [0 for _ in range(num_states * num_states)]

limit = 15 # 데이터의 생존 시간

# UAV가 실제 움직일 공간
world = np.zeros((num_states, num_states))
# world[len(world) // 2][len(world) // 2] = 1 # UAV의 초기 위치
world[len(world) // 2][len(world) // 2] = 1 # UAV의 초기 위치

# 학습 매개변수
gamma = 0.1  # 할인 계수
learning_rate = 0.01  # 학습률
num_episodes = 1000  # 총 에피소드 수
minus = 0.00001 # 입실론 감쇠값


epsilon = 0.9  # 입실론 값 설정

total_reward = 0
total_reward_= 0

# 환경에서 R-matrix 업데이트 함수
def update_R_matrix(location, action, loc):
    # UAV가 데이터가 소실되기 전에 센싱 지역에 도착해서 데이터를 전송할 경우

    global total_reward, total_reward_
    
    # 근데 데이터를 감시 못했으면 해당 상태에 대한 보상값을
    # 높여서 UAV가 자주 방문하게 해야 하지 않을까?
    if field[location[0] * num_states + location[1]][0] == 1 :

        #print('감지 및 제거 확인 요망')

        total_reward += 1
        R[loc[0] * num_states + loc[1]][action] += 1
        field[location[0] * num_states + location[1]][0] = 0
        field[location[0] * num_states + location[1]][1] = 0



def move(location, a) :
    # P는 1로 설정
    # 4방향으로 알파버전하고 추후 여유롭다 싶으면 8방향으로 수정

    # 동 서 남 북 : 0 1 2 3

    # 에이전트의 이동 횟수를 보기 위해서 주석처리
    #world[location[0]][location[1]] = 0

    if a == 0 : # 동
        if location[1] % num_states == num_states - 1 :
            pass
        else :
           location[1] += 1
 
    elif a == 1 : # 서
        if location[1] % num_states == 0 :
            pass
        else :
            location[1] -= 1
  
    elif a == 2 : # 남
        if location[0] >= num_states - 1 : 
            pass 
        else :
            location[0] += 1

    elif a == 3 : # 북
        if location[0] == 0 :
            pass
        else :
            location[0] -= 1
    
    elif a == 4:
        pass

    world[location[0]][location[1]] += 1

    return location

# ---

location = [len(world) // 2, len(world) // 2] # 최초 중앙의 좌표값 전달

# 각 상태들의 데이터 발생 확률
percents = [1.0, 0.9, 0.9, 1.0, 1.0, 1.0,
            1.0, 0.9, 0.9, 1.0, 1.0, 1.0,
            1.0, 0.9, 0.9, 1.0, 1.0, 1.0,
            1.0, 0.9, 0.9, 1.0, 1.0, 1.0,
            1.0, 0.9, 0.9, 1.0, 1.0, 1.0,
            1.0, 0.9, 0.9, 1.0, 1.0, 1.0]

def get_random_diff(last):
    while True:
        new_num = random.randint(0, 4)
        if new_num != last:
            return new_num
        
def start():
    global location, total_reward, total_reward_, world
    array = []
    array2 = []

    for j in range(20) :
        location = [len(world) // 2, len(world) // 2] # 최초 중앙의 좌표값 전달
        total_reward = 0
        total_reward_= 0
        # UAV가 실제 움직일 공간
        # world = np.zeros((num_states, num_states))
    
        world[len(world) // 2][len(world) // 2] = 1 # UAV의 초기 위치
        
        action = None
        
        # 입실론-그리디 전략에 따른 행동 선택을 위한 함수
        def choose_action(state, last):
            global epsilon, minus

            epsilon -= minus
            # print('epsilon : ', epsilon)
            
            if random.random() < epsilon:
                # print('무작위 행동 선택됨')
                while True:
                    new_action = random.randint(0, 3)
                    if new_action != last:
                        # new_action = int(input())
                        return new_action
            # 무작위 행동 선택
            else:
                max_value = np.max(Q[state, :])  # 최대값 찾기
                max_indices = np.where(Q[state, :] == max_value)[0]  # 최대값을 가진 모든 인덱스 찾기
                return random.choice(max_indices)  # 최대값 인덱스 중에서 무작위로 하나 선택

        for _ in range(2000):
            action = choose_action(location[0] * num_states + location[1], action)  # 입실론-그리디 전략에 따라 행동 선택
            now_locate = []
            now_locate.append(location[0])
            now_locate.append(location[1])

            location = move(location, action)  # 에이전트 위치 이동
            update_R_matrix(location, action, now_locate)  # R-matrix 업데이트

            # 센싱 데이터의 존재 여부 및 카운트 업데이트
            for k in range(len(percents)):
                if field[k][0] == 1:  # 센싱 데이터가 존재하는 경우
                    field[k][1] += 1  # 카운트 증가
                    
                    if field[k][1] > limit:  # 카운트가 특정 값을 넘으면
                        check_date[k] += 1

                        field[k][0] = 0  # 데이터 제거
                        field[k][1] = 0  # 카운트 초기화

                        # 데이터가 손실된 지역의 왼쪽에서의 R matrix값 증가
                        if k % num_states != 0 : 
                            R[k - 1][0] += 1

                        # 데이터가 손실된 지역의 오른쪽에서의 R matrix값 증가
                        if k % num_states != 4 :
                            R[k + 1][1] += 1

                        # 데이터가 손실된 지역의 위에서의 R matrix값 증가
                        if k > 4 : # i > 4
                            R[k - num_states][2] += 1
                        
                        # 데이터가 손실된 지역의 아래쪽에서의 R matrix값 증가

                        if k < 20 : # i < 20
                            R[k + num_states][3] += 1

                        total_reward_ -= 1  # 누적 보상 감소
                    
                percent = random.random()  # 매 회차마다 무작위 확률 생성
                if percent >= percents[k]:  # 확률에 따른 데이터 생성
                    field[k][0] = 1
                    check_date[k] += 1
                    # 데이터가 손실된 지역의 왼쪽에서의 R matrix값 증가
                    if k % num_states != 0 : 
                        R[k - 1][0] += 5

                    # 데이터가 손실된 지역의 오른쪽에서의 R matrix값 증가
                    if k % num_states != 4 :
                        R[k + 1][1] += 5

                        # 데이터가 손실된 지역의 위에서의 R matrix값 증가
                    if k > 4 : # i > 4
                        R[k - num_states][2] += 5
                    
                        # 데이터가 손실된 지역의 아래쪽에서의 R matrix값 증가

                    if k < 20 : # i < 20
                        R[k + num_states][3] += 5
            
                loc = now_locate[0] * num_states + now_locate[1]
                next_loc = location[0] * num_states + location[1]

                num1 = R[loc][action]
                num2 = gamma * np.max(Q[next_loc])
                num3 = Q[loc][action]

                Q[loc][action] = Q[loc][action] + learning_rate * (num1 + num2 - num3)

        array.append(total_reward)
        array2.append(total_reward_)
        print('-------------------------------------')
        print(j, '번째 에피소드')
        print(world)
        print('종료시점의 epsilon :', epsilon)
        print('이번 에피소드 :', total_reward)
        print('-------------------------------------')
        
        print('에피소드 별로 누적된 보상')
        print(array)
        print('에피소드 별로 감소된 값')
        print(array2)


def show_array(array) :
    for i in range(len(array)) :
        if i % num_states == 0 :
            print()
        print(round(array[i], 2), end = ' ')
    print()

def show_field() :
    for i in range(len(field)) :

        if i % num_states == 0 :
            print()

        print(field[i], end = ' ')

    print()

def show_matrix(matrix) :
    for i in range(len(matrix)) :
        if i % 5 == 0 :
            print('-----------------')
        print(np.round(matrix[i], 2))
        
if __name__ == "__main__":

    start()
