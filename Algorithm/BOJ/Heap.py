# 1927

import sys
import heapq


class MinHeap:
    
    def __init__(self, N):
        self.N = N
        self.heap = []
    
    def insertNum(self):
        self.num = int(sys.stdin.readline())
        
        return self.num
        
    def getMinHeap(self):        
        for _ in range(self.N):
            if self.insertNum() != 0:
                heapq.heappush(self.heap, self.num)
            else:
                if self.heap:
                    print(heapq.heappop(self.heap))
                else:
                    print(0)

minheap = MinHeap(int(input()))
minheap.getMinHeap()



# 11279

import sys
import heapq


class MaxHeap:
    
    def __init__(self, N):
        self.N = N
        self.heap = []
    
    def insertNum(self):
        self.num = int(sys.stdin.readline())
        
        return self.num
        
    def getMaxHeap(self):        
        for _ in range(self.N):
            if self.insertNum() != 0:
                heapq.heappush(self.heap, -self.num)
            else:
                if self.heap:
                    print(-1 * heapq.heappop(self.heap))
                else:
                    print(0)

minheap = MaxHeap(int(input()))
minheap.getMaxHeap()



# 11286

import sys
import heapq


class absHeap:
    
    def __init__(self, N):
        self.N = N
        self.heap = []
    
    def insertNum(self):
        self.num = int(sys.stdin.readline())
        
        return self.num
        
    def getabsHeap(self):        
        for _ in range(self.N):
            if self.insertNum() != 0:
                heapq.heappush(self.heap, (abs(self.num), self.num))
            else:
                if self.heap:
                    print(heapq.heappop(self.heap)[1])
                else:
                    print(0)

absheap = absHeap(int(input()))
absheap.getabsHeap()



# 1766

import heapq

class Workbook:

    def __init__(self, N, M):        
        self.N = N
        self.M = M

        self.numLists = [[] for _ in range(self.N + 1)]
        self.inDegree = [0 for _ in range(self.N + 1)]

        self.heap = []
        self.result = []

    def GraphInformation(self):
        for i in range(self.M):
            x, y = map(int, input().split())
            self.numLists[x].append(y)
            self.inDegree[y] += 1

    def makeFirstHeap(self):
        for i in range(1, self.N + 1):
            if self.inDegree[i] == 0:
                heapq.heappush(self.heap, i)

    def makeTopologicalSort(self):
        while self.heap:
            temp = heapq.heappop(self.heap)
            self.result.append(temp)

            for i in self.numLists[temp]:
                self.inDegree[i] -= 1

                if self.inDegree[i] == 0:
                    heapq.heappush(self.heap, i)
    
    def showResult(self):
        for i in self.result:
            print(i, end = ' ')

                    
N, M = map(int, input(). split()) 
workbook = Workbook(N, M) 
workbook.GraphInformation() 
workbook.makeFirstHeap() 
workbook.makeTopologicalSort()
workbook.showResult()
