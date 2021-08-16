#include <iostream>

using namespace std;



class Fibonacci
{
public:
	int getFibonacci(int n);
};



int Fibonacci::getFibonacci(int n) {
	
	if ( n >= 3 ) {
		return getFibonacci(n - 1) + getFibonacci(n - 2);
	}
	else if (n == 2) {
		return 1;
	}
	else if (n == 1) {
		return 1;
	}
	else {
		return 0;
	}
	
}


int main() {

	int n;
	cin >> n;

	Fibonacci fibonacci;
	cout << fibonacci.getFibonacci(n);
}
