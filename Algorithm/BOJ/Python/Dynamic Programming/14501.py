def solution(time_reward_lists):

    answer = list(map(lambda x: x[1], time_reward_lists))
    answer.append(0)


    for idx in range(len(time_reward_lists) - 1, -1, -1):    
        if time_reward_lists[idx][0] + idx > len(time_reward_lists):
            answer[idx] = answer[idx + 1]
        else:
            answer[idx] = max(answer[idx + 1], time_reward_lists[idx][1] + answer[idx + time_reward_lists[idx][0]])
                
    return print(answer[0])
        

n = int(input())
time_reward_lists = []
for _ in range(n):
    input_ = list(map(int, input().split()))
    time_reward_lists.append(input_)

solution(time_reward_lists)
