// ColorfulBoxesAndBalls
package main

import (
	"fmt"
)

func getMinimum(a, b int) int {
	if a >= b {
		return b
	} else {
		return a
	}
}

func getMaximum(a, b int) int {
	if a >= b {
		return a
	} else {
		return b
	}
}

type ColorfulBoxesAndBalls struct {
}

func (c ColorfulBoxesAndBalls) getMaximum(numRed, numBlue, onlyRed, onlyBlue, bothColors int) int {
	answer := -100000
	change := getMinimum(numRed, numBlue)

	for i := 0; i <= change; i++ {
		currentScore := (numRed-i)*onlyRed + (numBlue-i)*onlyBlue + 2*i*bothColors

		answer = getMaximum(answer, currentScore)
	}

	return answer
}

func main() {

	colorfulboxesandballs := ColorfulBoxesAndBalls{}
	fmt.Println(colorfulboxesandballs.getMaximum(2, 3, 100, 400, 200))
}
