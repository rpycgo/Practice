# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:43:05 2021

@author: MJH
"""


def is_number_in_array(array: list, target: int):
    result = 0
    
    left = 0
    right = len(array)-1
    
    while left <= right:
        mid = (left + right) // 2
        if array[mid] == target:
            result = 1
            break
        elif array[mid] > target:
            right = mid-1
        else:
            left = mid+1
            
    return result

def main() -> None:
    N = int(input())
    array = list(map(int, input().split()))
    array.sort()
    
    
    M = int(input())
    number_list_to_check_in_array = list(map(int, input().split()))
    
    for number in number_list_to_check_in_array:
        print(is_number_in_array(array, number))

if __name__ == '__main__':
    main()
    