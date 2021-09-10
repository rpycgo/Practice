// 9095
package main

import (
	"fmt"
)

func solution(n int) int {

	fibonacci := []int{0, 1, 2, 4}

	for idx := 4; idx < n+1; idx++ {
		fibonacci = append(fibonacci, (fibonacci[idx-3] + fibonacci[idx-2] + fibonacci[idx-1]))
	}

	return fibonacci[n]
}

func main() {

	var n int
	var input int
	fmt.Scanln(&n)

	for idx := 0; idx < n; idx++ {
		fmt.Scanln(&input)
		fmt.Println(solution(input))
	}

}
