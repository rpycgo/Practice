# 1065

class Hansu:
    
    def __init__(self):
        self.hansu = 0
        
    def insertNumber(self):
        self.N = int(input())
        
    def getHansu(self):
        for n in range(1, self.N + 1) :
            if n <= 99 :
                self.hansu += 1 
                
            else :     
                split_num = list(map(int, str(n)))
                
                if split_num[0] - split_num[1] == split_num[1] - split_num[2] :
                    self.hansu += 1
                    
        print(self.hansu)


hansu = Hansu()
hansu.insertNumber()
hansu.getHansu()
