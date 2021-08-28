// linked_memory
package main

import (
	"fmt"
)

type List struct {
	arr         [100]int
	curPosition int
	numOfData   int
}

func (list *List) initList() {
	list.curPosition = -1
	list.numOfData = 0
}

func (list *List) insertAtFirst(data int) {
	if list.numOfData >= 1 {
		for i := list.numOfData; i > 0; i-- {
			list.arr[i] = list.arr[i-1]
		}
	}

	list.arr[0] = data
	list.numOfData++
}

func main() {
	list := List{}
	list.initList()
	fmt.Println(list.arr[0])
	list.insertAtFirst(11)
	list.insertAtFirst(11)
	list.insertAtFirst(22)
	list.insertAtFirst(22)
	list.insertAtFirst(33)

	for index, value := range list.arr {
		fmt.Println(index, value)

		if value == 0 {
			break
		}
	}
}
