# 2309

import sys


class Dwarf():
    
    def __init__(self):
        self.Lists = []
        self.First = 0
        self.Second = 0
    
    def insertHeight(self):
        for _ in range(9):
            self.Lists.append(int(sys.stdin.readline()))
            
        self.sum = sum(self.Lists)
        
    def getAnswer(self):
        for i in range(8):
            for j in range(i + 1, 9):
                if self.sum - (self.Lists[i] + self.Lists[j]) == 100:
                    self.First = self.Lists[i]
                    self.Second = self.Lists[j]
                    
        self.Lists.remove(self.First)
        self.Lists.remove(self.Second)
        self.Lists.sort()
        
        for i in self.Lists:
            print(i)
 

dwarf = Dwarf()
dwarf.insertHeight()
dwarf.getAnswer()



# 1476

class YearCalculation:
    
    def __init__(self, E, S, M):        
        self.E = E
        self.S = S
        self.M = M
        self.YEAR = 1
    
    def CalculateYear(self):
        while True:
            if (self.YEAR - self.E) % 15 == 0 and (self.YEAR - self.S) % 28 == 0 and (self.YEAR - self.M) % 19 == 0:
                print(self.YEAR)
                break
            self.YEAR += 1

E, S, M = map(int, input().split())
yearcal = YearCalculation(E, S, M)
yearcal.CalculateYear()
