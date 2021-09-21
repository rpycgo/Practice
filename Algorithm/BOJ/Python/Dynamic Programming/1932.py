def solution(triangle):
    
    for i in range(1, len(triangle)):
        for j in range(i + 1):
            if j == 0:
                triangle[i][j] += triangle[i - 1][j]
            elif j == i:
                triangle[i][j] += triangle[i - 1][j - 1]
            else:
                triangle[i][j] += max(triangle[i - 1][j], triangle[i - 1][j - 1])    
                
    return print(max(triangle[-1]))



n = int(input())
triangle = []
for _ in range(n):
    row = list(map(int, input().split()))
    triangle.append(row)
    
solution(triangle)
