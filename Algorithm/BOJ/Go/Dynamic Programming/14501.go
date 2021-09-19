// 14051
package main

import (
	"bufio"
	"fmt"
	"os"
)


func getMax(a, b int) int {
	if ( a >= b ) {
		return a
	} else {
		return b
	}
}


func solution(time_reward_lists [][]int) int {

	answer := make([]int, len(time_reward_lists), len(time_reward_lists))
	answer = append(answer, 0)
	
	for idx := len(time_reward_lists) - 1; idx >= 0; idx-- {
		if ( time_reward_lists[idx][0] + idx > len(time_reward_lists) ) {
			answer[idx] = answer[idx + 1]
		} else {
			answer[idx] = getMax(answer[idx + 1], time_reward_lists[idx][1] + answer[idx + time_reward_lists[idx][0]])
		}
	}
	
	return answer[0]
}


func main() {

	var n int
	fmt.Scanln(&n)

	r := bufio.NewReader(os.Stdin)
	time_reward_lists := make([][]int, n, n)
	for i := 0; i < n; i++ {
		element := make([]int, 2, 2)
		for j := 0; j < 2; j++ {
			fmt.Fscan(r, &element[j])	
		}
		time_reward_lists[i] = element		
	}
	
	fmt.Println(solution(time_reward_lists))
}
