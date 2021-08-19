#include <iostream>

using namespace std;

class RecursiveSum
{
public:
	int sum(int n);
	int recursiveSum(int n);
};


int RecursiveSum::sum(int n)
{
	int ret = 0;
	for (int i = 1; i <= n; ++i)
	{
		ret += i;
	}

	return ret;
}


int RecursiveSum::recursiveSum(int n)
{
	if (n == 1) 
	{
		return 1;
	}

	return n + recursiveSum(n - 1);
}



int main()
{
	int n;
	cin >> n;

	RecursiveSum recursive_sum;
	cout << recursive_sum.sum(n) << endl;
	cout << recursive_sum.recursiveSum(n) << endl;
}
