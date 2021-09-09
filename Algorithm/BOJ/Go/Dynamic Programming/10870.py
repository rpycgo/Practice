// 10870
package main

import (
	"fmt"
)

func solution(n int) int {

	fibonacci := []int{0, 1}

	for idx := 2; idx < n+1; idx++ {
		fibonacci = append(fibonacci, (fibonacci[idx-2] + fibonacci[idx-1]))
	}

	return fibonacci[n]
}

func main() {

	var n int
	fmt.Scanln(&n)

	fmt.Println(solution(n))
}
