#include <iostream>

using namespace std;


int main()
{
	int price = 0;

	while (price != -1)
	{
		cout << "판매 금액을 만원 단위로 입력(-1 to end): ";
		cin >> price;

		if (price == -1)
		{
			cout << "프로그램을 종료합니다" << endl;
			return 0;
		}

		cout << "이번 달 급여: " << 50 + price * 0.12 << endl;
	}
}
