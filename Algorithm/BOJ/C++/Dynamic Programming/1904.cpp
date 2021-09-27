#include <iostream>
#include <vector>

using namespace std;

int solution(int N) {

	vector<int> answer = { 1, 2 };

	if (N == 1) {
		return answer[0];
	}
	else {
		for (int i = 2; i < N; i++) {
			answer.push_back((answer[1] + answer[0]) % 15746);
			answer.erase(answer.begin());
		}

		return answer[1];
	}
}


int main() {
	int N;
	cin >> N;

	cout << solution(N);
	
	return 0;
}
