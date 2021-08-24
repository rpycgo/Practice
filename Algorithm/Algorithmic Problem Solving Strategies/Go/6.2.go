// 6.2.go
package main

import (
	"fmt"
)

type Combination struct {
}

func (c Combination) pick(n int, picked []*int, toPick int) {
	if toPick == 0 {
		c.printPicked(picked)
		return
	}

	var smallest *int
	if len(picked) == 0 {
		*smallest = 0
	} else {
		*smallest = picked[(len(picked)-1)] + 1
	}

	for next := *smallest; next < n; next++ {
		picked.append(picked, next)
		c.pick(n, picked, toPick)
		picked = picked[:(len(picked) - 2)]
	}
}

func (c Combination) printPicked(picked []*int) {
	for i := 0; i < len(picked); i++ {
		fmt.Println(picked[i])
	}
	fmt.Println('\n')
}

func main() {
	combination := Combination{}
	a := []int{0, 1, 2}

	combination.pick(10, &a, 2)
}
