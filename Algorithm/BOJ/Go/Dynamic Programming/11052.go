// 11052
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	reader := bufio.NewReader(os.Stdin)

	var N int
	fmt.Fscanln(reader, &N)

	cards := make([]int, N+1)
	for i := 1; i <= N; i++ {
		fmt.Fscanf(reader, "%d ", &cards[i])
	}

	fmt.Println(getMaxPrice(N, &cards))
}

func getMaxPrice(N int, cards *[]int) int {
	answer := make([]int, N+1)

	for i := 1; i < N+1; i++ {
		for j := 1; j < i+1; j++ {
			answer[i] = getMax(answer[i], answer[i-j]+(*cards)[j])
		}
	}

	return answer[N]
}

func getMax(a int, b int) int {
	if a >= b {
		return a
	} else {
		return b
	}
}
