// 6.1
package main

import (
	"fmt"
)


type RecursiveSum struct {
}

func (r RecursiveSum) sum(n int) int {
	ret := 0

	for i := 0; i <= n; i++ {
		ret += i
	}

	return ret
}

func (r RecursiveSum) recursiveSum(n int) int {

	if n == 1 {
		return 1
	}

	return n + r.recursiveSum(n-1)
}



func main() {

	var n int
	fmt.Scanln(&n)

	recursive_sum := RecursiveSum{}
	fmt.Println(recursive_sum.sum(n))
	fmt.Println(recursive_sum.recursiveSum(n))
}
