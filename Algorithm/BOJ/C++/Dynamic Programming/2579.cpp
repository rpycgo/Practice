#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


int solution(vector<int> stairs, int n) {
	
	vector<int> answer;
	answer.push_back(stairs[0]);

	for (int i = 1; i < n; i++) {
		if (i == 1) {
			answer.push_back((answer[i - 1] + stairs[i]));
		}
		else if (i == 2) {
			answer.push_back(max((stairs[i] + stairs[i - 1]), (stairs[i] + answer[i - 2])));
		}
		else {
			answer.push_back(max(stairs[i] + stairs[i - 1] + answer[i - 3], stairs[i] + answer[i - 2]));
		}
	}
	
	return answer.back();
}


int main() {
	int n, input;
	vector<int> stairs;

	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> input;
		stairs.push_back(input);		
	}
					
	cout << solution(stairs, n);
}
