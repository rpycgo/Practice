#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


int solution(vector<int> num_array) {

	for (unsigned int i = 0; i < num_array.size() - 1; i++) {
		if (num_array[i + 1] <= num_array[i + 1] + num_array[i]) {
			num_array[i + 1] += num_array[i];
		}
	}

	return *max_element(num_array.begin(), num_array.end());
}


int main() {
	int n, input;
	vector<int> num_array;

	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> input;
		num_array.push_back(input);
	}

	cout << solution(num_array);

	return 0;
}
