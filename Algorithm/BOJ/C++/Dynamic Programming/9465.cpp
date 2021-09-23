#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


int solution(vector<vector<int>> stickers, int n) {

	if (n >= 2) {
		stickers[0][1] += stickers[1][0];
		stickers[1][1] += stickers[0][0];
	}

	for (int i = 2; i < n; i++) {
		stickers[0][i] = max(stickers[0][i] + stickers[1][i - 1], stickers[0][i] + stickers[1][i - 2]);
		stickers[1][i] = max(stickers[1][i] + stickers[0][i - 1], stickers[1][i] + stickers[0][i - 2]);
	}

	return max(stickers[0][n - 1], stickers[1][n - 1]);
}


int main() {
	int T, n;
	cin >> T;

	for (int i = 0; i < T; i++) {
		vector<vector<int>> stickers;
		cin >> n;

		for (int j = 0; j < 2; j++) {
			vector<int> sticker(n);

			for (auto& input : sticker) {
				cin >> input;				
			}
			stickers.push_back(sticker);
		}
		cout << solution(stickers, n) << endl;
	}

	return 0;
}
