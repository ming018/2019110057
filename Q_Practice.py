# Q러닝 간단 구현
import numpy as np

size = 3
field_len = size * size  # 필드 크기
action_count = 4  # 가능한 행동의 수

# Q-테이블 초기화
Q = np.zeros((field_len, action_count))

# Q-테이블 업데이트 함수
def update_Q_table(learning_rate=0.1, discount_factor=0.8):
    while True :
        for state in range(field_len):
            for action in range(action_count):
                # 현재 상태-행동 쌍에 대한 Q 값
                predict = Q[state, action]

                # 여기에서 새로운 상태와 보상을 정의합니다.
                # 실제 환경에서는 이 값들이 에이전트의 상호작용을 통해 결정됩니다.
                if state == 8 :
                    new_state = 8
                else :
                    new_state = (state + 1) % field_len  # 예시를 위한 새로운 상태
                
                if state == 5 :
                    reward = 1
                else :
                    reward = -1  # 예시를 위한 보상

                # 새로운 상태에서의 최대 Q 값
                target = reward + discount_factor * np.max(Q[new_state, :])

                # Q-테이블 업데이트
                Q[state, action] += learning_rate * (target - predict)
        
        print(Q)
        cmd = input()
        if cmd == ',' :
            break

# Q-테이블 업데이트 실행
update_Q_table()

# 업데이트된 Q-테이블 출력

