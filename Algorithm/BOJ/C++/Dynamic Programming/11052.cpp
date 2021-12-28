#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int getMaxPrice(int N, vector<int> *cards);

int main() {
	int N, price;
	vector<int> cards;

	cin >> N;
	
	cards.push_back(0);
	for (int i = 0; i < N; i++) {
		cin >> price;
		cards.push_back(price);
	}

	cout << getMaxPrice(N, &cards);
}

int getMaxPrice(int N, vector<int>* cards) {
	vector<int> answer(N + 1);

	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= i; j++) {
			answer[i] = max(answer[i], answer[i - j] + (*cards)[j]);
		}
	}

	return answer[N];
}
