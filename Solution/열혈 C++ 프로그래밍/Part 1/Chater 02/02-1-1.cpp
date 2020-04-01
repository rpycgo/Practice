#include <iostream>

using namespace std;


void increase(int& x)
{
	x++;
}

void reverse(int& x)
{
	x *= -1;
}


int main()
{
	int x = 1;

	increase(x);
	cout << x << endl;
	
	reverse(x);
	cout << x << endl;	
}
