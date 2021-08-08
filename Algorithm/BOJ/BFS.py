# 2178

import sys

class mazeProbe():
    
    def __init__(self, n, m):
        self.N = n
        self.M = m
        self.maze = []
        self.queue = []
        self.dx = [1, -1, 0, 0]
        self.dy = [0, 0, -1, 1]
    
    def insertLists(self):
        for _ in range(self.N):
            self.maze.append(list(sys.stdin.readline()))
        
    def getAnswer(self):
        self.queue = [[0, 0]]
        self.maze[0][0] = 1
        
        while self.queue:
            a, b = self.queue[0][0], self.queue[0][1]
            del self.queue[0]
            
            for i in range(4):
                x = a + self.dx[i]
                y = b + self.dy[i]
                
                if 0 <= x < self.N and 0 <= y < self.M and self.maze[x][y] == '1':
                    self.queue.append([x, y])
                    self.maze[x][y] = self.maze[a][b] + 1
         
        print(self.maze[self.N -1][self.M - 1])
        
        
n, m = map(int, sys.stdin.readline().split())
mazeprobe = mazeProbe(n, m)
mazeprobe.insertLists()
mazeprobe.getAnswer()



# 2667

import sys

class numberingComplex():
    
    def __init__(self, n):
        self.N = n
        self.cnt = 0
        self.complex = []
        self.num = []
        self.dx = [1, -1, 0, 0]
        self.dy = [0, 0, -1, 1]
    
    def insertLists(self):
        for _ in range(self.N):
            self.complex.append(list(sys.stdin.readline()))
                
    def dfs(self, a, b):
        self.complex[a][b] = '0'
        self.cnt += 1
        
        for i in range(4):
            x = a + self.dx[i]
            y = b + self.dy[i]
        
            if x < 0 or x >= self.N or y < 0 or y >= self.N:
                continue         
            if self.complex[x][y] == '1':
                self.dfs(x, y)
        
    def getAnswer(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.complex[i][j] == '1':
                    self.cnt = 0
                    self.dfs(i, j)
                    self.num.append(self.cnt)
         
        print(len(self.num))
        self.num.sort()
        for i in self.num:
            print(i)
        
                
numberingcomplex = numberingComplex(int(sys.stdin.readline()))
numberingcomplex.insertLists()
numberingcomplex.getAnswer()
