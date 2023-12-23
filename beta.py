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

limit = 10 # 데이터의 생존 시간

# UAV가 실제 움직일 공간
world = np.zeros((num_states, num_states))
# world[len(world) // 2][len(world) // 2] = 1 # UAV의 초기 위치
world[len(world) // 2][len(world) // 2] = 1 # UAV의 초기 위치

# 학습 매개변수
gamma = 0.01  # 할인 계수
learning_rate = 0.01  # 학습률
num_episodes = 10000  # 총 에피소드 수
minus = 0.00001 # 입실론 감쇠값

episode_count= 100 # 에피소드 진행 횟수

loss = 5 # UAV가 센싱 지역에 도착하기 전에 데이터가 전송될 경우의 보상 값
sending = 5 # UAV가 센싱 지역에 도착하여 데이터가 전송될 경우의 보상 값

epsilon = 0.9  # 입실론 값 설정

total_reward = 0 # 데이터를 보내기 전에 에이전트가 보내는 수치
total_reward_= 0 # 에이전트가 보내지 못한 수치

# 각 상태들의 데이터 발생하지 않을 확률
percents = [1.0, 0.9, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

check_date = [0 for _ in range(num_states * num_states)]

# 환경에서 R-matrix 업데이트 함수
def check_R_matrix(location):
#def check_R_matrix(location, loc):
#def update_R_matrix(location, action, loc):

    global total_reward, total_reward_ 
    
    # UAV가 데이터가 소실되기 전에 센싱 지역에 도착해서 데이터를 전송할 경우
    if field[location[0] * num_states + location[1]][0] == 1 :
        
        check_date[location[0] * num_states + location[1]] = 9

        total_reward += 1 # 
        # print(loc[0] * num_states + loc[1] + 1)
        
        # 해당 지역이 데이터가 좀 더 많이 생긴다는 학습을 위해 R의 값을 변경..?
        # R[loc[0] * num_states + loc[1]][action] += 5
        field[location[0] * num_states + location[1]][0] = 0
        field[location[0] * num_states + location[1]][1] = 0

        # 데이터가 센싱된 지역의 왼쪽에서의 R matrix값 증가
        if location[0] * num_states + location[1] % num_states != 0 : 
            R[location[0] * num_states + location[1] - 1][0] += loss

        # 데이터가 센싱된 지역의 오른쪽에서의 R matrix값 증가
        if location[0] * num_states + location[1] % num_states != 5 :
            R[location[0] * num_states + location[1]][1] += loss

        # 데이터가 센싱된 지역의 위에서의 R matrix값 증가
        if location[0] * num_states + location[1] > 4 : # i > 4
            R[location[0] * num_states + location[1] - num_states][2] += loss
        
        # 데이터가 센싱된 지역의 아래쪽에서의 R matrix값 증가
        if location[0] * num_states + location[1] < 20 : # i < 20
            R[location[0] * num_states + location[1] + num_states][3] += loss
        
    # 가장 자리에 있을 때 한자리에 계속 있지 않도록 보상 수정
    for i in range(len(R)) :
        if i < 6 :
            R[i][3] = 0

        if i % 6 == 0 :
            R[i][1] = 0

        if i % 6 == 5 :
            R[i][0] = 0

        if i > 29 :
            R[i][2] = 0

def move(location, a) :
    # P는 1로 설정

    # 동 서 남 북 : 0 1 2 3

    # 에이전트의 이동 횟수를 보기 위해서 주석처리
    # world[location[0]][location[1]] = 0

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



        
def start():
    global location, total_reward, total_reward_, world
    array = [] # 에피소드 마다 UAV가 센싱 지역에 도착해 데이터를 보냈을 경우 누적되는 수치
    array2 = [] # 에피소드 마다 UAV가 센싱 지역에 도착하지 못해 데이터를 보내지 못하는 경우 누적되는 수치

    for j in range(episode_count) : # 에피소드 진행 횟수
        location = [len(world) // 2, len(world) // 2] # 최초 중앙의 좌표값 전달

        total_reward = 0 # 이번 에피소드에서 UAV가 센싱 지역에 도착해 데이터를 보냈을 경우 누적되는 수치
        total_reward_= 0 # 이번 에피소드에서 UAV가 센싱 지역에 도착하지 못해 데이터를 보내지 못하는 경우 누적되는 수치


        # UAV가 실제 움직일 공간
        world = np.zeros((num_states, num_states))
    
        world[len(world) // 2][len(world) // 2] = 1 # UAV의 초기 위치

        # 필드에서 데이터가 전송 되지 못한 경우를 카운팅 하는 배열
        check_date = [0 for _ in range(num_states * num_states)]
        
        action = None
        
        # 입실론-그리디 전략에 따른 행동 선택을 위한 함수
        def choose_action(state, last): # 현재 상태, 선택된 액션
            global epsilon, minus

            epsilon -= minus # 입실론 감소

            if random.random() < epsilon: # 랜덤한 수치가 입실론 보다 적을 경우

                # 액션이 기존과 다른 액션이 나오도록 지정
                while True:
                    new_action = random.randint(0, 3)

                    if new_action != last:
                        return new_action
                    

            else: # 최적 정책 선택
                max_value = np.max(Q[state, :])  # 최대값 찾기
                max_indices = np.where(Q[state, :] == max_value)[0]  # 최대값을 가진 모든 인덱스 찾기
                return random.choice(max_indices)  # 최대값 인덱스 중에서 무작위로 하나 선택

        for _ in range(1000): # 에피소드 반복

            # 입실론-그리디 전략에 따라 행동 선택
            action = choose_action(location[0] * num_states + location[1], action) 

            now_locate = [] # 현재 위치를 담는 배열(y, x) 좌표
            now_locate.append(location[0])
            now_locate.append(location[1])

            location = move(location, action)  # 에이전트 위치 이동
            
            # update_R_matrix(location, action ,now_locate)  # R-matrix 업데이트
            #check_R_matrix(location, now_locate)  # UAV가 센싱 지역에 도착 했는지 확인
            check_R_matrix(location)  # UAV가 센싱 지역에 도착 했는지 확인

            # 센싱 데이터의 존재 여부 및 카운트 업데이트
            for k in range(len(percents)):
                if field[k][0] == 1:  # 센싱 데이터가 존재하는 경우
                    field[k][1] += 1  # 카운트 증가
                    
                    if field[k][1] > limit:  # 카운트가 특정 값을 넘으면(너무 오래 걸릴 경우)
                        check_date[k] += 1

                        field[k][0] = 0  # 데이터 제거
                        field[k][1] = 0  # 카운트 초기화

                        # 데이터가 손실된 지역의 왼쪽에서의 R matrix값 증가
                        if k % num_states != 0 : 
                            R[k - 1][0] += loss

                        # 데이터가 손실된 지역의 오른쪽에서의 R matrix값 증가
                        if k % num_states != 5 :
                            R[k + 1][1] += loss

                        # 데이터가 손실된 지역의 위에서의 R matrix값 증가
                        if k > 4 : # i > 4
                            R[k - num_states][2] += loss
                        
                        # 데이터가 손실된 지역의 아래쪽에서의 R matrix값 증가

                        if k < 20 : # i < 20
                            R[k + num_states][3] += loss

                        total_reward_ -= 1  # 누적 보상 감소
                else :   
                    percent = random.random()  # 매 회차마다 무작위 확률 생성
                    if percent >= percents[k]:  # 확률에 따른 데이터 생성
                        field[k][0] = 1

            # Q매트릭스 업데이트
            loc = now_locate[0] * num_states + now_locate[1]
            next_loc = location[0] * num_states + location[1]

            num1 = R[loc][action]
            num2 = gamma * np.max(Q[next_loc])
            num3 = Q[loc][action]

            Q[loc][action] = Q[loc][action] + learning_rate * (num1 + num2 - num3)

        # input()
        # show_field()
        # print('---------------')
        # print(world)

        array.append(total_reward)
        array2.append(total_reward_)
        print('-------------------------------------')
        print(j, '번째 에피소드')
        print(world)
        # print('종료시점의 epsilon :', epsilon)
        # print('이번 에피소드 :', total_reward)
        # print()
        # print('에이전트가 탐지 하지 못한 센싱 데이터')
        # show_array(check_date)
        
        print('-------------------------------------')
        
    print('에피소드 별로 누적된 보상')
    print(array)
    print('에피소드 별로 감소된 값')
    print(array2)
    show_matrix(Q)

        
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
        if i % 6 == 0 :
            print('-----------------')
        print(np.round(matrix[i], 2))
        
if __name__ == "__main__":

    start()
