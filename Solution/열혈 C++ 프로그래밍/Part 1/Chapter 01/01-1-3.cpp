#include <iostream>

using namespace std;


int main()
{
	int x;

	cout << "단 입력: ";
	cin >> x;

	for (int i = 1; i <= 9; i++)
	{
		cout << x << " * " << i << " = " <<  x * i << endl;
	}
}
