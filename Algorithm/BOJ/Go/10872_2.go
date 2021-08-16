// 10872
// struct refactoring
package main

import (
	"fmt"
)

type Factorial struct {
}

func (f Factorial) getFactorial(n int) int {
	if n >= 2 {
		return n * f.getFactorial(n-1)
	} else {
		return 1
	}
}

func main() {
	var n int
	fmt.Scanln(&n)

	factorial := Factorial{}
	fmt.Println(factorial.getFactorial(n))
}
