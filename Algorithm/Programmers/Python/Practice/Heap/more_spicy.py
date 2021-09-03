import heapq

def solution(scoville, K):
    answer = 0
    heapq.heapify(scoville)
    
    
    while True:    
        try:
            if scoville[0] >= K:
                return answer
            
            else:
                answer += 1
                min_first = heapq.heappop(scoville)
                min_second = heapq.heappop(scoville)
                
                mixed_scoville = min_first + (min_second) * 2
                heapq.heappush(scoville, mixed_scoville)
                
        except:
            return -1
