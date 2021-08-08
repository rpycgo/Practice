#2751

n = int(input())

solution = [int(input()) for _ in range(n)]

solution = sorted(solution)

for i in solution:
    print(i)

    
    
# 11650

n = int(input())

solution = [list(map(int, input().split())) for _ in range(n)]

solution.sort(key = lambda x: (x[0], x[1]))

for coord in solution:
    print(coord[0], coord[1])

    
    
# 11651

n = int(input())

solution = [list(map(int, input().split())) for _ in range(n)]

solution.sort(key = lambda x: (x[1], x[0]))

for coord in solution:
    print(coord[0], coord[1])
    
    
    
# 10814

n = int(input())

solution = [list(input().split()) for _ in range(n)]

solution.sort(key = lambda x: int(x[0]))

for value in solution:
    print(value[0], value[1])
    
    
    
# 10825

n = int(input())

solution = [list(input().split()) for _ in range(n)]

solution.sort(key = lambda x: (-int(x[1]), int(x[2]), -int(x[3]), x[0]))

for value in solution:
    print(value[0])
    

    
# 10989

import sys

n = int(sys.stdin.readline())

solution = [0] * 10001

for _ in range(n):
    solution[int(sys.stdin.readline())] += 1

for i in range(10001):
    sys.stdout.write('%s\n' % i * solution[i])

    
    
# 11652

from collections import Counter

n = int(input())

solution = [input() for _ in range(n)]

dic = Counter(solution)

dic_sorted = sorted(dic.items(), key = lambda item: (item[1], -int(item[0])))

print(int(dic_sorted[-1][0]))



# 11004

n, k = map(int, input().split())

solution = list(map(int, input().split()))

solution.sort()

print(solution[k - 1])



# 10828

# using stack but time over
class stack:
    
    def __init__(self):
        self.solution = []
        
    def push(self, x):
        self.solution.append(x)
    
    def pop(self):
        try:
            print(self.solution.pop())
        except:
            print(-1)
    
    def size(self):
        print(len(self.solution))
    
    def empty(self):
        if self.size() == 0:
            print(1)
        else:
            print(0)
    
    def top(self):
        try:
            print(self.solution[-1])
        except:
            print(-1)
        
    
n = int(input())
stack = stack()

for _ in range(n):
    cmd = input().split()
    
    if cmd[0] == 'push':
        stack.push(cmd[1])
    elif cmd[0] == 'pop':
        stack.pop()
    elif cmd[0] == 'size':
        stack.size()
    elif cmd[0] == 'empty':
        stack.empty()
    elif cmd[0] == 'top':
        stack.top()

        
        
# 9012

class parenthesis:
    
    def __init__(self):
        self.solution = ''
        self.sum = 0
    
    def enter(self, x):
        self.solution = x
      
N = int(input())

for _ in range(N):    
    
    early_stop = 0
    paren = parenthesis()
    input_list = input()    
    paren.enter(input_list)
    
    for i in paren.solution:
    
        if i == '(':
            paren.sum += 1
        elif i == ')':
            paren.sum -= 1
        
        if paren.sum < 0:
            print('NO')
            early_stop = 1
            break
        
    if early_stop == 0:
        if paren.sum == 0:
            print('YES')
        else:
            print('NO')
