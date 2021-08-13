// 1065
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	var n int
	input := bufio.NewReader(os.Stdin)
	fmt.Fscanln(input, &n)

	var count = get_Hansu(n)
	fmt.Println(count)
}

func get_Hansu(number int) int {

	var count int

	if number < 100 {
		count = number
		return count
	}

	for i := 100; i <= number; i++ {
		units := i % 10
		tens := i / 10 % 10
		hundreds := i / 100

		if hundreds-tens == tens-units {
			count++
		}
	}

	count += 99

	return count
}
