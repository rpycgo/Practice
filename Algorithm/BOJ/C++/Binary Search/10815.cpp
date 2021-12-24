#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

bool isNumberInHoldingCardList(vector<int>* holding_card_list, int number_to_check);

int main() {
	int N, M, input_number;

	cin >> N;
	vector<int> holding_card_list;
	for (int i = 0; i < N; i++) {
		cin >> input_number;
		holding_card_list.push_back(input_number);
	}
	sort(holding_card_list.begin(), holding_card_list.end());

	cin >> M;
	vector<int> card_list_to_check;
	for (int i = 0; i < M; i++) {
		cin >> input_number;
		card_list_to_check.push_back(input_number);
	}

	for (int number_to_check : card_list_to_check) {
		cout << isNumberInHoldingCardList(&holding_card_list, number_to_check) << " ";
	}

}


bool isNumberInHoldingCardList(vector<int>* holding_card_list, int number_to_check) {
	bool result = false;

	int left = 0;
	int right = holding_card_list->size() - 1;
	int mid;

	while (left <= right) {
		mid = (left + right) / 2;

		if (holding_card_list->at(mid) == number_to_check) {
			result = true;
			break;
		}
		else if (holding_card_list->at(mid) > number_to_check) {
			right = mid - 1;
		}
		else {
			left = mid + 1;
		}
	}

	return result;
}