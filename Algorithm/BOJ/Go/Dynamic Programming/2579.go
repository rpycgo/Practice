// 2579
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
)


func solution(stairs []int, n int) int {

	answer := make([]int, n, n)
	answer[0] = stairs[0]
	

	for i := 1; i < n; i++ {
		if i == 1 {
			answer[i] = (answer[i-1] + stairs[1])
		} else if i == 2 {
			max := int(math.Max(float64(stairs[i]+stairs[i-1]), float64(stairs[i]+answer[i-2])))
			answer[i] = max
		} else {
			max := int(math.Max(float64(stairs[i]+stairs[i-1]+answer[i-3]), float64(stairs[i]+answer[i-2])))
			answer[i] = max
		}
	}
	
	return answer[n-1]
}


func main() {

	var n int
	fmt.Scanln(&n)

	r := bufio.NewReader(os.Stdin)
	stairs := make([]int, n, n)
	for i := 0; i < n; i++ {
		fmt.Fscan(r, &stairs[i])
	}
	
	fmt.Println(solution(stairs, n))
}
