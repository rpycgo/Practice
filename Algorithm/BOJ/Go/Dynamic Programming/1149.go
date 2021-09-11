// 1149
package main

import (
	"fmt"
	"math"
)

func solution(rgb [][]int, n int) int {

	for i := 1; i < n; i++ {
		rgb[i][0] = int(math.Min(float64(rgb[i-1][1]), float64(rgb[i-1][2])) + float64(rgb[i][0]))
		rgb[i][1] = int(math.Min(float64(rgb[i-1][0]), float64(rgb[i-1][2])) + float64(rgb[i][1]))
		rgb[i][2] = int(math.Min(float64(rgb[i-1][0]), float64(rgb[i-1][1])) + float64(rgb[i][2]))
	}

	return int(math.Min(math.Min(float64(rgb[n-1][0]), float64(rgb[n-1][1])), float64(rgb[n-1][2])))
}

func main() {

	var n int
	fmt.Scanln(&n)

	rgb := [][]int{}
	for i := 0; i < n; i++ {
		var a, b, c int
		fmt.Scan(&a, &b, &c)
		rgb = append(rgb, []int{a, b, c})
	}

	fmt.Println(solution(rgb, n))
}
