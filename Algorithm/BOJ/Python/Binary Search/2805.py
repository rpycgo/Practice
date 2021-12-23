def main():
    _, M = map(int, input().split())
    trees = list(map(int, input().split()))
    
    find_max_height(trees, M)

def find_max_height(trees: list, target_log: int):
    start = 1
    end = max(trees)
    
    while start <= end:
        mid = (start + end) // 2        
        log = 0
        
        for tree in trees:
            if tree > mid:
                log += tree - mid
        
        if log >= target_log:
            start = mid + 1
        else:
            end = mid - 1
            
    print(end)
            
if __name__ == '__main__':
    main()