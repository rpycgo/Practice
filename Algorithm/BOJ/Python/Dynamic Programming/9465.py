def solution(stickers, n):
    
    if n >= 2:
        stickers[0][1] += stickers[1][0]
        stickers[1][1] += stickers[0][0]
    
    for i in range(2, n):
        stickers[0][i] = max(stickers[0][i] + stickers[1][i - 1], stickers[0][i] + stickers[1][i - 2])
        stickers[1][i] = max(stickers[1][i] + stickers[0][i - 1], stickers[1][i] + stickers[0][i - 2])
        
    return print(max(list(map(lambda x: x[-1], stickers))))



T = int(input())
for _ in range(T):
    stickers = []
    n = int(input())
    for _ in range(2):        
        sticker = list(map(int, input().split()))
        stickers.append(sticker)
        
    solution(stickers, n)
