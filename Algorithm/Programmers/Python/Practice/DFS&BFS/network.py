from collections import deque


class solution:
    def __init__(self):
        self.visited = ''
        self.computers = ''
        self.n = ''
    
    
    def __call__(self, n, computers, method = 'bfs'):
        self.n = n
        self.computers = computers
        self.visited = [0 for _ in range(len(self.computers))]

        answer = 0 
        
        
        for i in range(n):
            if not self.visited[i]:
                if method == 'bfs':
                    self.bfs(i)
                    answer += 1
                elif method == 'dfs':
                    self.bfs(i)
                    answer += 1
                else:
                    raise(ValueError)
            
        return answer


    def dfs(self, i):
        self.visited[i] = 1
        for j in range(self.n):
            if self.computers[i][j] and not self.visited[j]:
                self.dfs(j)
                
                
    def bfs(self, i):        
        queue = deque()
        queue.append(i)
        while queue:
            i = queue.popleft()
            self.visited[i] = 1
            for j in range(self.n):
                if self.computers[i][j] and not self.visited[j]:
                    queue.append(j)
