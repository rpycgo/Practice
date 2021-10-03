// 1929
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
)

func find_prime(M, N int) {

	for number := M; number < N+1; number++ {
		if number == 1 {
			continue
		}
		if (number != 2) && (number%2 == 0) {
			continue
		}

		max_check_num := int(math.Pow(float64(number), 1.0/2.0)) + 1
		i := 1
		for i < max_check_num {
			if (i != 1) && (number%i == 0) {
				break
			}
			i++
		}
		if i == max_check_num {
			fmt.Println(number)
		}
	}
}

func main() {

	var M, N int
	r := bufio.NewReader(os.Stdin)

	fmt.Fscan(r, &M)
	fmt.Fscan(r, &N)
	find_prime(M, N)
}
