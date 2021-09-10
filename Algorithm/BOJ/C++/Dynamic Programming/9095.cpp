#include <iostream>
#include <vector>

using namespace std;


int solution(int n) {
	vector<int> answer = { 0, 1, 2, 4 };

	for (int i = 4; i < n + 1; i++) {
		answer.push_back((answer[i - 3] + answer[i - 2] + answer[i - 1]));
	}

	return answer[n];
}


int main() {
	int n, input;

	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> input;
		cout << solution(input) << endl;
	}

}
