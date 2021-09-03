def solution(money):
    
    answer1 = [0] * len(money)
    answer1[0] = money[0]
    answer1[1] = max(money[0], money[1])
    
    for i in range(2, len(money) - 1):
        answer1[i] = max(answer1[i - 1], money[i] + answer1[i - 2])
    
    
    answer2 = [0] * len(money)
    answer2[0] = 0
    answer2[1] = money[1]
    
    for i in range(2, len(money)):
        answer2[i] = max(answer2[i - 1], money[i] + answer2[i - 2])
    
    
    return max(max(answer1), max(answer2))
