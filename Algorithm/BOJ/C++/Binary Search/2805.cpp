#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

long FindMaxHeight(vector<int>* trees, int target_log);

int main() {
	int N, M, i, tree;

	cin >> N >> M;
	vector<int> trees;

	for (i = 0; i < N; i++) {
		cin >> tree;
		trees.push_back(tree);
	}

	cout << FindMaxHeight(&trees, M);
}

long FindMaxHeight(vector<int>* trees, int target_log) {
	int left = 1;
	int right = *max_element(trees->begin(), trees->end());
	int mid;

	while (left <= right) {
		mid = (left + right) / 2;
		long log_sum = 0;

		for (int tree : *trees) {
			if (tree > mid) {
				log_sum += tree - mid;
			}
		}

		if (log_sum >= target_log) {
			left = mid + 1;
		}
		else {
			right = mid - 1;
		}
	}
	return right;
}