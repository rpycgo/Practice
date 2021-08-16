// 10870
package main

import (
	"fmt"
)

type Fibonacci struct {
}

func (f Fibonacci) getFibonacci(n int) int {

	if n >= 3 {
		return f.getFibonacci(n-1) + f.getFibonacci(n-2)
	} else if n == 2 {
		return 1
	} else if n == 1 {
		return 1
	} else {
		return 0
	}
}

func main() {
	var n int
	fmt.Scanln(&n)

	fibonacci := Fibonacci{}
	fmt.Println(fibonacci.getFibonacci(n))
}
