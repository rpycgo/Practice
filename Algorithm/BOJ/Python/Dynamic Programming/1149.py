def solution(rgb, n):
    
    for i in range(1, n):
        rgb[i][0] = min(rgb[i - 1][1], rgb[i -1][2]) + rgb[i][0]
        rgb[i][1] = min(rgb[i - 1][0], rgb[i -1][2]) + rgb[i][1]
        rgb[i][2] = min(rgb[i - 1][0], rgb[i -1][1]) + rgb[i][2]
        
    return print(min(rgb[n - 1][0], rgb[n - 1][1], rgb[n - 1][2]))



n = int(input())
rgb = []
for _ in range(n):
    rgb.append(list(map(int, input().split())))
solution(rgb, n)
