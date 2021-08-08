# 11654

n = input()
print(ord(n))



# 1152

class getWordCount():
    
    def __init__(self, input):
        self.input = input.strip()
        
    def CalWordCount(self):
        
        split_string = self.input.split(' ')
        num = len(split_string)
        print(num)
        
    
getwordcount = getWordCount('Teullinika Teullyeotzi ')
getwordcount.CalWordCount()
