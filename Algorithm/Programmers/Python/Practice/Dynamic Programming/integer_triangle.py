def solution(triangle):
    triangle_length = len(triangle)
    
    if triangle_length == 1:
        return triangle[0][0]

    else:
        for i in range(triangle_length):
            if i == 0:
                pass
            
            else:
                for j in range(i + 1):
                    if j == 0:
                        triangle[i][0] += triangle[i - 1][0]
                    elif j == i:
                        triangle[i][-1] += triangle[i - 1][-1]
                    else:
                        triangle[i][j] += max(triangle[i - 1][j - 1], triangle[i - 1][j])
    
    return max(triangle[i])
