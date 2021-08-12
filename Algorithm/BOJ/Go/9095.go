// 9095.go
package main

import (
	"fmt"
)

func solution(n int) {

	array := [3]int{1, 2, 4}

	if n <= 3 {
		return array[(n - 1)]
	} else if n > 3 {
		return solution(n-1) + solution(n-2) + solution(n-3)
	}
}

func main() {

	var T int
	var n int
	fmt.Scanln(&T)

	for i := 0; i < T; i++ {
		n = fmt.Scanln(&n)
		fmt.Println(solution(n))
	}

}
