#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


int solution(vector<vector<int>> triangle) {

	int n = triangle.size();

	for (int i = 1; i < n; i++) {
		for (int j = 0; j < i + 1; j++) {
			if (j == 0) {
				triangle[i][j] += triangle[i - 1][j];
			}
			else if (j == i) {
				triangle[i][j] += triangle[i - 1][j - 1];
			}
			else {
				triangle[i][j] += max(triangle[i - 1][j], triangle[i - 1][j - 1]);
			}						
		}
	}

	return *max_element(triangle[n - 1].begin(), triangle[n - 1].end());
}


int main() {
	int n;
	vector<vector<int>> triangle;

	cin >> n;
	for (int i = 0; i < n; i++) {
		vector<int> element((i + 1));
		for (auto& input : element) {
			cin >> input;
		}
		triangle.push_back(element);
	}

	cout << solution(triangle);

	return 0;
}
