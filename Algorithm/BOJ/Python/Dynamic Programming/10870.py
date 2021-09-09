def solution(n):
    
    fibonacci = [0, 1]
    
    for idx in range(2, n + 1):
        fibonacci.append((fibonacci[idx - 2] + fibonacci[idx - 1]))
        
    return fibonacci[n]
        

n = int(input())
print(solution(n))
