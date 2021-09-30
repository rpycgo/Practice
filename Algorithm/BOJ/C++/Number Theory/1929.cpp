#include <iostream>
#include <math.h>

using namespace std;

void find_prime(int M, int N) {

	for (int number = M; number < N + 2; number++) {
		if (number == 1) {
			continue;
		}

		int max_check_num = int(sqrt(number)) + 1;
		int i = 1;
		while (i < max_check_num) {			
			if ((i != 1) && (number % i == 0)) {
				break;
			}
			i++;
		}
		if (i == max_check_num) {
			cout << number << endl;
		}
	}
}


int main() {
	int M, N;
	cin >> M >> N;

	find_prime(M, N);
	
	return 0;
}
