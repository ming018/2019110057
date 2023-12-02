import random

percents = [ 0 for _ in range(4) ]
field = [[0, 0] for q in range(4)]

percent = random.random()


percents[0] = 0.4
percents[1] = 0.6
percents[2] = 0.8
percents[3] = 0.9

reward = 0

while True :

    for i in range(len(percents)) :
        if field[i][0] == 1 :
            field[i][1] += 1
            if field[i][1] > 3 :
                field[i][0] = 0
                field[i][1] = 0
                reward -= 1
            
            print('요기 지나감')
            continue
        
        if percent >= percents[i] :
            field[i][0] = 1
    

    print('percent : ', percent)
    print(field[: len(field) // 2])
    print(field[len(field) // 2 :])
    print(reward)
    cmd = input()

    if cmd == '.' :
        break

    percent = random.random()