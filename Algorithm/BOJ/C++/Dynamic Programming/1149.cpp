#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


int solution(vector<vector<int>> rgb, int n) {
	
	for (int i = 1; i < n; i++) {
		rgb[i][0] = min(rgb[i - 1][1], rgb[i - 1][2]) + rgb[i][0];
		rgb[i][1] = min(rgb[i - 1][0], rgb[i - 1][2]) + rgb[i][1];
		rgb[i][2] = min(rgb[i - 1][0], rgb[i - 1][1]) + rgb[i][2];
	}

	return *min_element(rgb[n - 1].begin(), rgb[n - 1].end());
}


int main() {
	int n, input;
	vector<vector<int>> rgb;

	cin >> n;
	for (int i = 0; i < n; i++) {
		vector<int> element;
		for (int j = 0; j < 3; j++) {
			cin >> input;
			element.push_back(input);
		}
		rgb.push_back(element);
	}
					
	cout << solution(rgb, n);
}
