#include <iostream>
#include <vector>

using namespace std;

int solution(int N) {

	vector<int> answer = { 1, 2 };

	for (int i = 2; i < N; i++) {
		answer.push_back((answer[i - 1] + answer[i - 2]) % 15746);
	}

	return answer[N - 1];
}


int main() {
	int N;
	cin >> N;

	cout << solution(N);
	
	return 0;
}
