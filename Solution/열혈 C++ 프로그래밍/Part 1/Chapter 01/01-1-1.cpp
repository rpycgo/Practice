#include <iostream>

using namespace std;


int main()
{
	int sum = 0, x = 0;	

	for (int i = 1; i <= 5; i++)
	{
		cout << i << "번째 정수 입력: ";
		cin >> x;
		sum += x;
	}

	cout << "합계: " << sum << endl;
}
