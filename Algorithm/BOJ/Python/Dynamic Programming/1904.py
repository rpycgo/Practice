def solution(N):
    
    answer = [1, 2]
    
    if N == 1:
        print(answer[0])
        
    else:
        for _ in range(2, N):
            answer.append( (answer[1] + answer[0]) % 15746)
            answer.pop(0)
        
        return print(answer[1])



N = int(input())
solution(N)
