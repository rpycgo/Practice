#include <iostream>

using namespace std;



class Factorial
{
public:
	int getFactorial(int n);
};



int Factorial::getFactorial(int n) {
	
	if ( n >= 2 ) {
		return n * getFactorial(n - 1);
	}
	else {
		return 1;
	}
	
}


int main() {

	int n;
	cin >> n;

	Factorial factorial;
	cout << factorial.getFactorial(n);
}
