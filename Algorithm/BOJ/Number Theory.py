# 10610

class Thirty():
    
    def __init__(self, n):
        self.N = n
        
    def getMax(self):
        self.N.sort(reverse = True)
        sum = 0
        
        for i in self.N:
            sum += int(i)
        
        if sum % 3 != 0 or '0' not in self.N:
            print(-1)
        else:
            print(''.join(self.N))

        
thirty = Thirty(list(input()))
thirty.getMax()
