# 11728

class arrayJoin:
    
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.solution = []
        
    def joinList(self, x):
        self.solution = self.solution + x
        

N, M = map(int, input().split())
sol = arrayJoin(N, M)
sol.joinList(list(map(int, input().split())))
sol.joinList(list(map(int, input().split())))
sol.solution.sort()
[print(i, end = ' ') for i in sol.solution]
