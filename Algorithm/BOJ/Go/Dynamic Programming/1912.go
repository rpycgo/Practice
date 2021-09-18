// 1912
package main

import (
	"bufio"
	"fmt"
	"os"
)

func getMax(num_array []int) int {

	max := num_array[0]

	for i := 1; i < len(num_array); i++ {
		if num_array[i] > max {
			max = num_array[i]
		}
	}

	return max
}

func solution(num_array []int) int {

	for i := 0; i < len(num_array)-1; i++ {
		if num_array[i+1] <= num_array[i+1]+num_array[i] {
			num_array[i+1] += num_array[i]
		}
	}

	return getMax(num_array)
}

func main() {

	var n int
	fmt.Scan(&n)

	r := bufio.NewReader(os.Stdin)
	num_array := make([]int, n, n)
	for i := 0; i < n; i++ {
		fmt.Fscan(r, &num_array[i])
	}

	fmt.Println(solution(num_array))
}
