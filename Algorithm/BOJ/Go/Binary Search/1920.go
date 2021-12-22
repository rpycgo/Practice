// 1920
package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
)

func main() {
	reader := bufio.NewReader(os.Stdin)

	var N, M int
	fmt.Fscanln(reader, &N)

	array := make([]int, N)

	for i := 0; i < N; i++ {
		fmt.Fscanf(reader, "%d ", &array[i])
	}
	sort.Ints(array)

	fmt.Fscanln(reader, &M)
	var number_list_to_check_in_array = make([]int, M)
	for i := 0; i < M; i++ {
		fmt.Fscanf(reader, "%d ", &number_list_to_check_in_array[i])
	}

	for _, target := range number_list_to_check_in_array {
		fmt.Println(isNumberInArray(&array, target))
	}

}

func isNumberInArray(array *[]int, target int) int {
	result := 0
	left := 0
	right := len(*array) - 1

	for left <= right {
		mid := (left + right) / 2

		if (*array)[mid] == target {
			result = 1
			break
		} else if (*array)[mid] > target {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	return result
}
