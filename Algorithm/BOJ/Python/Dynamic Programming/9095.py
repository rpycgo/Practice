def solution(n):
    answer = [0, 1, 2, 4]
    
    for idx in range(4, n + 1):
        answer.append((answer[idx -3] + answer[idx -2] + answer[idx -1]))
    
    return print(answer[n])


n = int(input())
for _ in range(n):
    n = int(input())
    solution(n)
