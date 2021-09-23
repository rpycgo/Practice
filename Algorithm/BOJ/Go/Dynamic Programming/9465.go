// 9465
package main

import (
	"bufio"
	"fmt"
	"os"
)

func getMax(a, b int) int {
	if a >= b {
		return a
	} else {
		return b
	}
}

func solution(stickers [][]int, n int) int {

	if n >= 2 {
		stickers[0][1] += stickers[1][0]
		stickers[1][1] += stickers[0][0]
	}

	for i := 2; i < n; i++ {
		stickers[0][i] = getMax(stickers[0][i]+stickers[1][i-1], stickers[0][i]+stickers[1][i-2])
		stickers[1][i] = getMax(stickers[1][i]+stickers[0][i-1], stickers[1][i]+stickers[0][i-2])
	}

	return getMax(stickers[0][n-1], stickers[1][n-1])
}

func main() {

	var T, n int
	r := bufio.NewReader(os.Stdin)
		
	fmt.Fscan(r, &T)

	for i := 0; i < T; i++ {
		stickers := make([][]int, 2)
		fmt.Fscan(r, &n)

		for j := 0; j < 2; j++ {
			sticker := make([]int, n, n)

			for k := 0; k < n; k++ {
				fmt.Fscan(r, &sticker[k])
			}
			
			stickers[j] = sticker
		}

		fmt.Println(solution(stickers, n))
	}
}
