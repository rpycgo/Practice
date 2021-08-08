# 1620

import sys


class PocketmonMaster():
    
    def __init__(self, n, m):
        self.N = n
        self.M = m
        
        self.pocketmonLists = []
        self.pocketmonDictionarys = {}
        
        
    def getListsandDictionarys(self):
        for i in range(self.N):
            pocketmon = sys.stdin.readline().strip()
            self.pocketmonLists.append(pocketmon)
            self.pocketmonDictionarys[pocketmon] = i + 1
    
    def getAnswer(self):
        for _ in range(self.M):
            inputs = sys.stdin.readline().strip()
        
            if inputs.isdigit():
                print(self.pocketmonLists[int(inputs) - 1])
            else:
                print(self.pocketmonDictionarys[inputs])
            

n, m = map(int, sys.stdin.readline().split())
pocketmons = PocketmonMaster(n, m)
pocketmons.getListsandDictionarys()
pocketmons.getAnswer()
