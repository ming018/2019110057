import numpy as np
import random

# field = np.zeros((5, 5)) # 5 * 5  사이즈의 필드 생성
field = np.zeros((3, 3)) # 3 * 3  사이즈의 연습용 필드 생성
actions = [0, 1, 2, 3] # 동 서 남 북

Q = np.zeros((len(field) * len(field[0]), len(actions))) # Q 매트릭스 생성
# R = np.zeros((len(field), len(actions)))

goal = [len(field) - 1, len(field) - 1]

R = [[0, -1, 0, -1], [0, 0, -1, -1], [-1, 0, 0, -1], # 연습용 R 매트릭스
        [-1, -1, 0, 0], [0, 0, 0, 0], [-1, -1, 100, 0],
        [0, -1, -1, 0], [100, 0, -1, -1], [0, 0, 0, 0]]

Gamma = 0.8
epsilon = 0.9

# alpha
learning_rate = 0.1

# Q(state, action) = Q(state, action) + alpha* [R(state, action) + Gamma * max(Q(next state, all actions)) - Q(state, action)]

for z in range(10) :
    for i in range(len(Q) - 1) :
        for a in actions :
             Q[i][a] = Q[i][a] + learning_rate * (R[i][a] + Gamma * max(Q[i + 1]) - Q[i][a])
    print(z,'번째 실행의 Q')
    print(Q)
    print('------------------')


agent = [0, 0]

def next_state(agent, action) : 
        if action == 0 and agent[1] < len(field) : # 동
                agent[1] += 1

        if action == 1 and agent[1] > 0 : # 서
                agent[1] -= 1

        if action == 2 and agent[0] < len(field[0]) : # 남
                agent[0] += 1

        if action == 3 and agent[0] > 0 : # 북
                agent[0] -= 1

        return agent, action

        
if random.random() <= epsilon :
        agent, action = next_state(agent, random.randint(0, 4))

else :
        agent, action = next_state(agent, np.argmax(Q[agent[0] * 3 + agent[1]]))



print(agent)





# for i in range(len(field)) :
#       for k in range(len(field[0])) :
#            field[i][k] = np.argmax(Q[i * 3 + k])

# print(field)
# action = []

# for _ in range(len(field)) :
#     for p in range(len(actions)) :
#         for i in range(len(field)) :
#                 for a in actions :
#                         action.append(Q[0][a])
        

# Q[0][0] = R[0][0] + Gamma * max(action)

