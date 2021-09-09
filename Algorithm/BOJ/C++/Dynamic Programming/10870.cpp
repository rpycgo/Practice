#include <iostream>
#include <vector>

using namespace std;


int solution(int n) {
	vector<int> fibonacci = { 0, 1 };

	for (int idx = 2; idx < n + 1; idx++) {
		fibonacci.push_back((fibonacci[idx - 2] + fibonacci[idx - 1]));
	}

	return fibonacci[n];
}



int main() {
	int n;
	cin >> n;
	
	cout << solution(n);

	return 0;
}
