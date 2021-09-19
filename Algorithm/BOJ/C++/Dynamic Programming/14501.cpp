#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


int solution(vector<vector<int>> time_reward_lists) {

	vector<int> answer;	
	for (const auto& element: time_reward_lists) {
		answer.push_back(element[1]);
	}
	answer.push_back(0);

	for (int idx = time_reward_lists.size() - 1; idx >= 0; idx--) {
		if (time_reward_lists[idx][0] + idx > time_reward_lists.size()) {
			answer[idx] = answer[idx + 1];
		} 
		else {
			answer[idx] = max(answer[idx + 1], time_reward_lists[idx][1] + answer[idx + time_reward_lists[idx][0]]);
		}
	}

	return answer[0];
}


int main() {
	int n, input;
	vector<vector<int>> time_reward_lists;

	cin >> n;
	for (int i = 0; i < n; i++) {
		vector<int> element(2);
		for (auto& input : element) {
			cin >> input;
		}
		time_reward_lists.push_back(element);
	}

	cout << solution(time_reward_lists);

	return 0;
}
