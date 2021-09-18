def solution(num_array):

    for i in range(len(num_array) - 1):
        if num_array[i + 1] <= num_array[i] + num_array[i + 1]:
            num_array[i + 1] = num_array[i] + num_array[i + 1]   
    
    return print(max(num_array))
        

n = int(input())
num_array = list(map(int, input().split()))
solution(num_array)
