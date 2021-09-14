// 2579
package main

import (
	"fmt"
	"math"
)

func solution(stairs []int, n int) int {

	answer := []int{stairs[0]}

	for i := 1; i < n; i++ {
		if i == 1 {
			answer = append(answer, (answer[i-1] + stairs[1]))
		} else if i == 2 {
			max := int(math.Min(float64(stairs[i]+stairs[i-1]), float64(stairs[i]+answer[i-2])))
			answer = append(answer, max)
		} else {
			max := int(math.Min(float64(stairs[i]+stairs[i-1]+answer[i-3]), float64(stairs[i]+answer[i-2])))
			answer = append(answer, max)
		}
	}

	return answer[n-1]
}


func main() {

	var n int
	fmt.Scan(&n)

	stairs := []int{}
	for i := 0; i < n; i++ {
		var input int
		fmt.Scan(&input)
		stairs = append(stairs, input)
	}

	fmt.Println(solution(stairs, n))
}
