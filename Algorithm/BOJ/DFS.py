# 1260

class DFSBFS():
    
    def __init__(self, n, m):
        self.N = n
        self.M = m
        self.s = [[0] * (self.N + 1) for i in range(self.N + 1)]
        self.visit = [0 for i in range(self.N + 1)]
        
    def insertCoordinate(self):
        for _ in range(self.M):
            x, y = map(int, input().split())
            self.s[x][y] = 1
            self.s[y][x] = 1
            
    def DFS(self, v):
        print(v, end = ' ')
        self.visit[v] = 1
        
        for i in range(1, self.N + 1):
            if self.visit[i] == 0 and self.s[v][i] == 1:
                self.DFS(i)
    
    def BFS(self, v):
        queue = [v]
        self.visit[v] = 0
        
        while(queue):
            v = queue[0]
            print(v, end = ' ')
            del queue[0]            
        
            for i in range(1, self.N + 1):
                if self.visit[i] == 1 and self.s[v][i] == 1:
                    queue.append(i)
                    self.visit[i] = 0
           
n, m, v = map(int, input().split())
dfsbfs = DFSBFS(n, m)
dfsbfs.insertCoordinate()
dfsbfs.DFS(v)
print()
dfsbfs.BFS(v)
