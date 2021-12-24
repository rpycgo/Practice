// 10815
package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	writer := bufio.NewWriter(os.Stdout)
	defer writer.Flush()

	var N, M int

	fmt.Fscanln(reader, &N)
	holding_card_list := make([]int, N)
	for i := 0; i < N; i++ {
		fmt.Fscanf(reader, "%d ", &holding_card_list[i])
	}
	sort.Ints(holding_card_list)

	fmt.Fscanln(reader, &M)
	card_list_to_check := make([]int, M)
	for i := 0; i < M; i++ {
		fmt.Fscanf(reader, "%d ", &card_list_to_check[i])
	}

	for i := 0; i < M; i++ {
		fmt.Fprintf(writer, "%d ", isNumberInHoldingCardList(&holding_card_list, card_list_to_check[i]))
	}
}

func isNumberInHoldingCardList(holding_card_list *[]int, number_to_check int) int {
	result := 0

	left := 0
	right := len(*holding_card_list) - 1
	var mid int

	for left <= right {
		mid = (left + right) / 2

		if (*holding_card_list)[mid] == number_to_check {
			result = 1
			break
		} else if (*holding_card_list)[mid] > number_to_check {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return result
}
