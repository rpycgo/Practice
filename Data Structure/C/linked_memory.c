#include <stdio.h>

#define True  1
#define False 0
#define LIST_LEN 100
typedef int LData;

typedef struct __ArrayList {
	LData arr[LIST_LEN];
	int curPosition;
	int numOfData;
} ArrayList;

typedef ArrayList List;


void initList(List* plist) {
	(plist->curPosition) = -1;
	(plist->numOfData) = 0;
}

void insertAtFirst(List* plist, LData data) {
	if (plist->numOfData >= 1) {
		for (int i = plist->numOfData; i > 0; i--) {
			plist->arr[i] = plist->arr[i - 1];
		}
	}
	
	plist->arr[0] = data;
	
	(plist->numOfData)++;
}

int getListLength(List* plist) {
	return plist->numOfData;
}

int main(void) {
	List list;
	int data;
	initList(&list);

	insertAtFirst(&list, 11);
	insertAtFirst(&list, 11);
	insertAtFirst(&list, 22);
	insertAtFirst(&list, 22);
	insertAtFirst(&list, 33);

	printf("Saved Data: %d \n", getListLength(&list));
	for (int i = 0; i < 5; i++) {
		printf("%d\n", list.arr[i]);
	}
}
