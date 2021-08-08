# Hash
## 1
'''fail to satisfy efficiency'''
def solution(participant, completion):
    
    if len(participant) > 1:
        for i in completion:
            participant.remove(i)
    
    answer = participant[0]
    
    return answer

---------------------------------------------------------------

from collections import Counter

def solution(participant, completion):
    
    if len(participant) > 1:
        cnt1 = Counter([x for x in participant])
        cnt2 = Counter([x for x in completion])
        answer = list(cnt1 - cnt2)[0]
    
    return answer


## 2
def solution(phone_book):
    
    phone_book.sort(key = lambda x : len(x))
    
    i = 0
    while(i < len(phone_book)):
        test_phone_book = phone_book
        test = phone_book[i]
        test_length = len(test)
        test_phone_book.remove(test)
        condition = test in list(map(lambda x : x[:test_length], test_phone_book))
        
        if condition == True:
            break;
        else:
            i += 1
    
    if condition == True:
        return False
    else:
        return True
    
    
## 3
from collections import Counter

def solution(clothes):
    
    answer = 1
    
    cnt = Counter([x[1] for x in clothes])
    for i in cnt.values():
        answer *= (i + 1)
        
    answer -= 1
    
    return answer


## 4
def solution(genres, plays):
    
    items = [i for i in enumerate(zip(genres, plays))]    
    
    genres_cnt = dict()
    for i in list(set(genres)):
        genres_cnt[i] = 0        
    for i in items:
        genres_cnt[i[1][0]] += i[1][1]
                
    items.sort(key = lambda x: (-genres_cnt[x[1][0]], -x[1][1]))
    
    cnt = [1]
    for i in range(1, len(items)):
        if items[i - 1][1][0] == items[i][1][0]:
            cnt.append(cnt[i - 1] + 1)
        else:
            cnt.append(1)
            
    idx = [i for i in range(len(cnt)) if cnt[i] <= 2]
    
    answer = [items[i][0] for i in idx]
        
    return answer




# Stack/Queue
## 1
def solution(heights):
    
    answer = [0]    
    
    for i in range(1, len(heights)):
        index = 0
        for j in range(i - 1, -1, -1):
            if heights[i] < heights[j]:
                index = j + 1
                break
        answer.append(index)
        
    return answer


# 3
from collections import Counter

def solution(progresses, speeds):
    
    diff = list(map(lambda x, y: (100 - x) / y, progresses, speeds))
    time = list(map(lambda x: int(x) if x - int(x) == 0.0 else int(x) + 1, diff))
    
    for i in range(len(time) - 1):
        if time[i] > time[i + 1]:
            time[i + 1] = time[i]
    
    count = Counter(time)
    answer = list(count.values())
    
    return answer


# 6
def solution(prices):
    
    answer = []
        
    for i in range(len(prices)):
        time = 0
        for j in range(i + 1, len(prices)):            
            time += 1
            if prices[i] > prices[j]:
                break
        answer.append(time)
                        
    return answer




# Heap
## 1  
'''fail to satisfy efficiency'''
def solution(scoville, K):
    
    cnt = 0
    scoville.sort()
    
    if len(scoville) == 0:
        return -1
    elif K == 0:
        return cnt
    if len(scoville) <= 1:
        if scoville[0] >= K:
            return cnt
        else:
            return -1
    else:
        while scoville[0] < K:
            cnt += 1
            scoville.append(scoville[0] + 2 * scoville[1])
            scoville.pop(0)
            scoville.pop(0)
            scoville.sort()            
            if len(scoville) == 1 and scoville[0] < K:
                return -1
            
    return cnt

---------------------------------------------------------------

import heapq

def solution(scoville, K):
    
    cnt = 0
    heapq.heapify(scoville)
    
    if len(scoville) == 0:
        return -1
    elif K == 0:
        return cnt
    if len(scoville) <= 1:
        if scoville[0] >= K:
            return cnt
        else:
            return -1
    else:
        while scoville[0] < K:
            cnt += 1
            min0 = heapq.heappop(scoville)
            min1 = heapq.heappop(scoville)
            heapq.heappush(scoville, min0 + 2 * min1)            
            if len(scoville) == 1 and scoville[0] < K:
                return -1
            
    return cnt


## 2
import heapq

def solution(stock, dates, supplies, k):
    
    answer, idx = 0, 0
    pq = []
    
    while stock < k:
        for i in range(idx, len(dates)):
            if stock < dates[i]:
                break
            heapq.heappush(pq, -supplies[i])
            idx = i + 1
        
        stock += heapq.heappop(pq) * -1
        answer += 1
    
    return answer

    
    
    
# Sort
## 1
def solution(array, commands):

    answer = []
    temp = 0
    
    for i in range(len(commands)):
        temp = array[commands[i][0] - 1:commands[i][1]]
        temp.sort()
        answer.append(temp[commands[i][2] - 1])
    
    return answer


## 2
def solution(numbers):
    
    numbers = list(map(lambda x: str(x), numbers))
    numbers.sort(key = lambda x: x * 3, reverse = True)
    
    answer = str(int(''.join(numbers)))
    
    return answer


## 3
def solution(citations):
            
    l = len(citations)
    h = 0
    
    while True:        
        upper_cit = sum(list(map(lambda x: x >= h, citations)))
        inf_cit = sum(list(map(lambda x: x <= h, citations)))
        
        if upper_cit >= h and (l - h) <= inf_cit:
            break
        else:
            h += 1
        
    return h




# Brute-Force Search
## 1
def solution(answer):
    
    student = [[1, 2, 3, 4, 5],
               [2, 1, 2, 3, 2, 4, 2, 5],
               [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]]
    
    
    student_answer = [[], [], []]
   
    for i in range(0, 3):
        quotient, remainder = divmod(len(answer), len(student[i]))
        student_answer[i] = student[i] * quotient + student[i][ : remainder]
        
        
    result = [[], [], []]
    
    for i in range(0, 3):        
        result[i] = [i + 1, sum(list(map(lambda j: student_answer[i][j] == answer[j], range(len(answer)))))]
        
    
    sum_max = max(map(lambda x: x[:][1], result))    
    answer = []

    for i in range(len(result)):
        if result[i][1] == sum_max:
            answer.append(i + 1)
   
    
    return(answer)


## 2
import math
from itertools import permutations


def solutions(numbers):
    
    length = len(numbers)
    permutation = []
    
    for i in range(length):
        temp = permutations(numbers, i + 1)
        permutation.append([''.join(i) for i in temp])
        
    permutation = sum(permutation, [])
    permutation = list(set(permutation))
        
    permutation = list(filter(lambda x: int(x[-1]) % 2 != 0, permutation))
    permutation = list(filter(lambda x: int(x[-1]) % 5 != 0, permutation))
    permutation = list(filter(lambda x: sum([int(i) for i in x]) % 3 != 0 if int(x) > 3 else x, permutation))
    
    permutation = list(map(lambda x: int(x), permutation))
    
    permutation = list(set(permutation))
    
    if 0 in permutation:
        permutation.remove(0)
    if 1 in permutation:
        permutation.remove(1)
        
    temp1 = len(permutation)
    temp2 = 0
    
    while temp1 != temp2:
        temp1 = len(permutation)
        for i in permutation:
            if(i >= 10):
                j = 1
                while(j <= math.sqrt(i) + 1):
                    j += 1
                    if(i % j == 0):
                        permutation.remove(i)
                        break
                
        temp2 = len(permutation)
    
    answer = len(permutation)
    
    
## 3

def solution(brown, yellow):
    
    x = (brown + 4 + ((brown + 4) ** 2 - 16 * (brown + yellow)) ** 0.5) / 4
    y = (brown + yellow) // x
   
    return [max(x, y), min(x, y)]
    



# Greedy
## 1
def solution(n, lost, reserve):
    
    vec = [1] * n
    
    for i in lost:
        vec[i - 1] -= 1
    for i in reserve:
        vec[i - 1] += 1
        
    for i in range(len(vec) - 1):
        if vec[i] == 0 and vec[i + 1] == 2:
            vec[i] += 1
            vec[i + 1] -= 1
            
    for i in range(len(vec) - 1, 0, -1):
        if vec[i] == 0 and vec[i - 1] == 2:
            vec[i] += 1
            vec[i - 1] -= 1
                     
    answer = sum(list(map(lambda x: x >= 1, vec)))
    
    return answer


## 3
'''test10 time over'''
def solution(number, k):
    
    answer = ''
    n = len(number) - k
    start = 0
    
    for i in range(0, n):
        _max = number[start]
        idx_max = start
        
        for j in range(start, k + i + 1):
            if _max < number[j]:
                _max = number[j]
                idx_max = j
                
        start = idx_max + 1
        answer += _max
                    
    return answer


## 4
def solution(people, limit):
    
    people.sort()
    
    start = 0
    end = len(people) - 1
    dup = 0
    
    while start < end:        
        if people[start] + people[end] <= limit:
            dup += 1
            start += 1
            end -= 1
        else:
            end -= 1
     
    answer = len(people) - dup
    
    return answer




# Dynamic Programming

## 2
def solution(triangle):
    
    for i in range(1, len(triangle)):
        for j in range(i + 1):
            if j == 0:
                triangle[i][j] += triangle[i - 1][j]
            elif j == i:
                triangle[i][j] += triangle[i - 1][j - 1]
            else:
                triangle[i][j] += max(triangle[i - 1][j], triangle[i - 1][j - 1])    
                
    return max(triangle[-1])




# DFS/BFS

##1
def solution(numbers, target):
    
    tree = [0] 
    
    for i in numbers:    
        sub_tree = []
        
        for j in tree:
            sub_tree.append(j + i)
            sub_tree.append(j - i)
            
        tree = sub_tree
        
    return tree.count(target)
