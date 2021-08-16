// 10872
package main

import (
	"fmt"
)

func getFactorial(n int) int {
	if n >= 2 {
		return n * getFactorial(n-1)
	} else {
		return 1
	}
}

func main() {
	var n int
	fmt.Scanln(&n)

	fmt.Println(getFactorial(n))
}
