#include <iostream>

using namespace std;


template <typename T>
T SumArray(T arr[], int len)
{
	T sum = 0;
	for (int i = 0; i < len; i++)
		sum += arr[i];

	return sum;
}


int main()
{
	int arr1[] = { 1, 2, 3, 4, 5 };
	cout << SumArray<int>(arr1, 5) << endl;

	float arr2[] = { 1.1, 2.1, 3.1, 4.1, 5.1 };
	cout << SumArray<float>(arr2, 5) << endl;
}
