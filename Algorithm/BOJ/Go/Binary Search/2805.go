// 2805
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	var N, M int
	fmt.Fscanln(reader, &N, &M)

	trees := make([]int, N)
	for i := 0; i < N; i++ {
		fmt.Fscanf(reader, "%d ", &trees[i])
	}

	fmt.Println(FindMaxHeight(&trees, M))
}

func GetMaxElement(array *[]int) int {
	max := -1

	for _, element := range *array {
		if element > max {
			max = element
		}
	}
	return max
}

func FindMaxHeight(trees *[]int, target_log int) int {
	left := 1
	right := GetMaxElement(trees)
	var mid int

	for left <= right {
		mid = (left + right) / 2
		log := 0

		for _, tree := range *trees {
			if tree > mid {
				log += (tree - mid)
			}
		}

		if log >= target_log {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return right
}
