def solution(stairs, n):
    
    answer = stairs[:1]
    
    for i in range(1, n):
        if i == 1:
            answer.append(answer[i - 1] + stairs[1])
        elif i == 2:
            answer.append(max(stairs[i] + stairs[i - 1], stairs[i] + answer[i - 2]))
        else:
            answer.append(max(stairs[i] + stairs[i - 1] + answer[i - 3], stairs[i] + answer[i - 2]))

    return print(answer[-1])



n = int(input())
stairs = []
for _ in range(n):
    input_ = int(input())
    stairs.append(input_)
solution(stairs, n)
