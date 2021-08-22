#include <iostream>
#include <vector>

using namespace std;


class Combination
{
public:
	void pick(int n, vector<int>& picked, int toPick);
	void printPicked(vector<int>& picked);
};



void Combination::pick(int n, vector<int>& picked, int toPick) {
	if (toPick == 0) {
		Combination::printPicked(picked);
		return;
	}

	int smallest = picked.empty() ? 0 : picked.back() + 1;

	for (int next = smallest; next < n; ++next) {
		picked.push_back(next);
		pick(n, picked, toPick - 1);
		picked.pop_back();
	}
}


void Combination::printPicked(vector<int>& picked) {
	for (unsigned int i = 0; i < picked.size(); i++) {
		cout << picked[i] << ' ';
	}
	cout << '\n';
}



int main() {
	Combination combination;
	vector<int> a = { 0, 1, 2 };

	combination.pick(10, a, 2);
}
