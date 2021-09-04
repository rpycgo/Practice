from collections import deque


def solution(begin, target, words):
    
    if target not in words:
        return 0
    
    queue = deque()
    queue.append((begin, 0))
    
    while queue:        
        changed_word, depth = queue.pop()
        
        for word in words:
            difference = 0
        
            for i in range(len(word)):
                if changed_word[i] != word[i]:
                    difference += 1
            
            if (difference == 1) and (word == target):      
                depth += 1
                return depth
        
            elif difference == 1:
                queue.appendleft((word, depth + 1))
    
    return 0
