// 9461
package main

import (
	"fmt"
)

func solution(n int) int {

	answer := []int{1, 1, 1, 2, 2}

	if n <= len(answer) {
		return answer[n-1]
	} else {
		for i := len(answer); i < n; i++ {
			answer = append(answer, (answer[i-1] + answer[i-5]))
		}
	}

	return answer[n-1]
}

func main() {

	var n int
	fmt.Scan(&n)

	for i := 0; i < n; i++ {
		var N int
		fmt.Scan(&N)
		fmt.Println(solution(N))
	}
}
