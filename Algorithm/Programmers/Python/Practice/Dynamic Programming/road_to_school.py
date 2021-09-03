import numpy as np


def solution(m, n, puddles):
    
    answer = np.ones([n, m], dtype = int).tolist()
    
    if len(puddles) >= 1:
        for puddle in puddles:
            x, y = puddle            
            answer[y - 1][x - 1] = 0            
    
    # column check
    for idx, x in enumerate(answer[0]):        
        if x == 0:            
            break
    for i in range(len(answer[0])):
        if i > idx:
            answer[0][i] = 0
    
    # row check
    row = list(map(lambda x: x[0], answer))
    for idx, x in enumerate(row):
        if x == 0:
            break
    for i in range(len(row)):
        if i > idx:
            answer[i][0] = 0
        
        
    
    for i in range(1, n):
        for j in range(1, m):
            if answer[i][j] != 0:
                answer[i][j] = answer[i][j - 1] + answer[i - 1][j]
    
    return answer[-1][-1] % 1000000007
