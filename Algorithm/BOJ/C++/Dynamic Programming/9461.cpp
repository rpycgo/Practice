#include <iostream>
#include <vector>

using namespace std;


long long int solution(int n) {
	
	vector<long long int> answer = { 1, 1, 1, 2, 2 };

	if (n <= answer.size()) {
		return answer[n - 1];
	} else {
		for (int i = answer.size(); i < n; i++) {
			answer.push_back(answer[i - 1] + answer[i - 5]);
		}
	}

	return answer.back();
}


int main() {
	int n, input;

	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> input;
		cout << solution(input) << endl;		
	}

	return 0;
}
