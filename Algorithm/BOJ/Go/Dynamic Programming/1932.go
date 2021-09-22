// 1932
package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
)

func getMax(a, b int) int {
	if a >= b {
		return a
	} else {
		return b
	}
}

func solution(triangle [][]int) int {

	n := len(triangle)

	for i := 1; i < n; i++ {
		for j := 0; j < i+1; j++ {
			if j == 0 {
				triangle[i][j] += triangle[i-1][j]
			} else if j == i {
				triangle[i][j] += triangle[i-1][j-1]
			} else {
				triangle[i][j] += getMax(triangle[i-1][j], triangle[i-1][j-1])
			}
		}
	}

	sort.Ints(triangle[n-1])

	return triangle[n-1][n-1]
}

func main() {

	var n int
	fmt.Scanln(&n)

	r := bufio.NewReader(os.Stdin)
	triangle := make([][]int, n, n)
	for i := 0; i < n; i++ {
		element := make([]int, (i + 1), (i + 1))
		for j := 0; j < (i + 1); j++ {
			fmt.Fscan(r, &element[j])
		}
		triangle[i] = element
	}

	fmt.Println(solution(triangle))
}
