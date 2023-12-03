import random
import numpy as np

Time_limit = 5 # 데이터가 살아 있을 시간
Size = 5
full_size = Size * Size

percents = [ 0 for _ in range(Size * Size) ]
field = [[0, 0] for q in range(Size * Size)]
world = [0 for _ in range(Size * Size)]

Q = np.zeros((full_size, 4))

first = len(world) // 2
world[first] = 1

def move(location, a) :
    # P는 1로 설정
    # 4방향으로 알파버전하고 추후 여유롭다 싶으면 8방향으로 수정


    # 동 서 남 북 : 0 1 2 3
    if a == 0 : # 동
        if location % Size == 4 :
            pass
        location[location] = 0

        location += 1

        world[location] = 1
       
    
    elif a == 1 : # 서
        if location % Size == 0 :
            pass
        
        location[location] = 0

        location -= 1

        world[location] = 1
        
    
    elif a == 2 : # 남
        if location >= 20 :
            pass
        
        location[location] = 0

        location += Size

        world[location] = 1

    elif a == 3 : # 북
        if location >= 4 :
            pass
        location[location] = 0

        location -= Size

        world[location] = 1

    return location

percent_ = 0.5 # 일단 초기 확률을 0.5로 잡고
for i in range(len(percents)) : # 각 상태별로 다른 확률을 지급
    percents[i] = percent_
    percent_ += 0.05

    if percent_ >= 0.8 :
        percent_ = 0.4


def show_percent() :
    for i in range(len(percents)) :
        
        
        if i % Size == 0 :
            print()

        print(round(percents[i], 2), end = ' ')

def show_field() :
    for i in range(len(field)) :
        
        
        if i % Size == 0 :
            print()

        print(field[i], end = ' ')

def start() : # percent를 매게 변수로 넣는 거로 수정
    total_reward = 0
    while True :
        percent = random.random() # 매 회차 확률이 변함
        for i in range(len(percents)) : # 각 위치에 데이터가 있는지 확인
            if field[i][0] == 1 : # 만약 센싱데이터가 있는 경우
                field[i][1] += 1 # 카운트 증가
                if field[i][1] > Time_limit : # 카운트가 특정 값(오래 있는 시간)을 넘을 경우
                    field[i][0] = 0 # 데이터를 없앰
                    field[i][1] = 0    

                    total_reward -= 1 # 보상을 낮춤
                
                continue
            
            if percent >= percents[i] : # 각각의 상태가 특정 확률을 넘으면
                field[i][0] = 1
        
        print()
        print('percent : ', percent)
        show_field()
        print(total_reward)
        cmd = input()

        if cmd == '.' :
            break

def learning(lr = 0.01, f = 0.8) : # lr = learning_rate = 0.8, f = discount_factor = 0.8
    while True :
        for state in range(full_size):
            for action in range(4):
                # 현재 상태-행동 쌍에 대한 Q 값
                predict = Q[state, action]

                # 여기에서 새로운 상태와 보상을 정의합니다.
                # 실제 환경에서는 이 값들이 에이전트의 상호작용을 통해 결정됩니다.
                new_state = (state + 1) % full_size  # 예시를 위한 새로운 상태
                
                reward = -1  # 예시를 위한 보상

                # 새로운 상태에서의 최대 Q 값
                target = reward + f * np.max(Q[new_state, :])

                # Q-테이블 업데이트
                Q[state, action] += lr * (target - predict)
        
        print(Q)
        cmd = input()
        if cmd == '.' :
            break

learning()