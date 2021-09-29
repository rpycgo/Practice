import math

def find_prime(M, N):
    
    for number in range(M, N + 1):
        if number == 1:
            continue
        if number != 2 and number % 2 == 0:
            continue
        
        max_check_num = int(math.sqrt(number)) + 1
        for i in range(1, max_check_num):
            if i != 1 and number % i == 0:
                break
        else:
            print(number)


M, N = map(int, input().split())
find_prime(M, N)
