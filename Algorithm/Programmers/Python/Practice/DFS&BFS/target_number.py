from collections import deque


def solution(numbers, target):
    
    queue = deque()
    
    queue.append([numbers[0],0])
    queue.append([-1*numbers[0],0])

    while queue:
        number, idx = queue.popleft()
        idx += 1
        if idx < len(numbers):
            queue.append([number+numbers[idx], idx])
            queue.append([number-numbers[idx], idx])
        else:
            break
    
    answer = sum(map(lambda x: x[0] == target, queue))

    return answer
