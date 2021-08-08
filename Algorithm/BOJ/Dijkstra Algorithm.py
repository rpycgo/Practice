# 1753

from heapq import heappush, heappop

class ShortestPath:
    
    def __init__(self):
        self.INF = 99999999
        self.V = 0
        self.E = 0
        self.K = 0
        self.nodes_lists = []
        self.heap = []
    
    def insertVE(self, v, e):
        self.V = v
        self.E = e
        self.points = [self.INF] * (self.V + 1)
    
    def insertK(self, k):
        self.K = k
    
    def Dijkstra(self, start):
