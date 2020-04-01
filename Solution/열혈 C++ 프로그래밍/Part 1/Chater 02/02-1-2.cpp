#include <iostream>

using namespace std;


void SwapPointer(int *&x, int *&y)
{
	int *temp = x;
	x = y;
	y = temp;
}


int main()
{
	int num1 = 5, num2 = 10;
	int *ptr1 = &num1, *ptr2 = &num2;

	cout << "[" << *ptr1 << "," << *ptr2 << "]" << endl;
	
	SwapPointer(ptr1, ptr2);
	cout << "[" << *ptr1 << "," << *ptr2 << "]" << endl;
}
