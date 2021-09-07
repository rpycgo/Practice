 def solution(n, times):
	answer = 0
	times.sort(key = lambda x: x[1])
	start = -1
	
	for time in times:
		if start < time[0]:
			answer += 1
			start = time[1]
			
	return answer


times = []
n = int(inputs())
for _ in range(n):
  times.append(list(map(int, input().split())))
  
solution(n, times)
