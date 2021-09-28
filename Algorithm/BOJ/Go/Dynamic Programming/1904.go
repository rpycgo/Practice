// 1904
package main

import (
	"bufio"
	"fmt"
	"os"
)

func solution(N int) int {

	answer := []int{1, 2}

	if N == 1 {
		return answer[0]
	} else {
		for i := 2; i < N; i++ {
			answer = append(answer, ((answer[1] + answer[0]) % 15746))
			answer = answer[1:]
		}

		return answer[1]
	}
}

func main() {

	var N int
	r := bufio.NewReader(os.Stdin)

	fmt.Fscan(r, &N)

	fmt.Println(solution(N))

}
