def solution(n):
    
    answer = [1, 1, 1, 2, 2]
    
    if n <= len(answer) - 1:
        return answer[n - 1]
    else:
        for i in range(len(answer), n):
            answer.append(answer[i - 1] + answer[i - 5])
    
    return print(answer[-1])


n = int(input())
for _ in range(n):    
    solution(int(input()))
