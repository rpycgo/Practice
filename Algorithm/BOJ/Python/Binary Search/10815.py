def main():
    N = int(input())
    holding_card_list = list(map(int, input().split()))
    holding_card_list.sort()
    
    N = int(input())
    card_list_to_check = list(map(int, input().split()))
    
    for number_to_check in card_list_to_check:
        print(is_number_in_holding_card_list(holding_card_list, number_to_check), end=' ')

def is_number_in_holding_card_list(holding_card_list:list, number_to_check: int) -> int:
    result = 0
    
    left = 0
    right = len(holding_card_list) - 1
    
    while left <= right:
        mid = (left + right) // 2        
        
        if holding_card_list[mid] == number_to_check:
            result = 1
            break
        elif holding_card_list[mid] > number_to_check:
            right = mid - 1
        else:
            left = mid + 1
            
    return result
            
if __name__ == '__main__':
    main()