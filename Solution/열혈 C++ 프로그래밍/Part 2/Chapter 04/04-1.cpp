#include <iostream>

using namespace std;


class FruitSeller
{
private:
	int APPLE_PRICE, numOfApples, myMoney;

public:
	void InitMembers(int price, int num, int money);
	int SaleApples(int money);
	void ShowSalesResult() const;
};

class FruitBuyer
{
	int myMoney, numOfApples;

public:
	bool InitMembers(int money);
	void BuyApples(FruitSeller& seller, int money);
	void ShowBuyResult() const;
};


void FruitSeller::InitMembers(int price, int num, int money)
{
	APPLE_PRICE = price;
	numOfApples = num;
	myMoney = money;
}

int FruitSeller::SaleApples(int money)
{
	if (money < 0)
	{
		cout << "0보다 작은 수를 입력할 수 없습니다" << endl;
		return false;
	}

	int num = money / APPLE_PRICE;
	numOfApples -= num;
	myMoney += money;

	return num;
}

void FruitSeller::ShowSalesResult() const
{
	cout << "남은 사과: " << numOfApples << endl;
	cout << "판매 수익: " << myMoney << endl << endl;
}



bool FruitBuyer::InitMembers(int money)
{
	if (money < 0)
	{
		cout << "0보다 작은 수를 입력할 수 없습니다" << endl;
		return false;
	}

	myMoney = money;
	numOfApples = 0;

	return true;
}

void FruitBuyer::BuyApples(FruitSeller &seller, int money)
{
		numOfApples += seller.SaleApples(money);
		myMoney -= money;
}

void FruitBuyer::ShowBuyResult() const
{
	cout << "현재 잔액: " << myMoney << endl;
	cout << "사과 개수: " << numOfApples << endl;
}


int main(void)
{
	FruitSeller seller;
	seller.InitMembers(1000, 20, 0);
	FruitBuyer buyer;
	buyer.InitMembers(5000);
	buyer.BuyApples(seller, 2000);

	cout << "과일 판매자의 현황" << endl;
	seller.ShowSalesResult();
	cout << "과일 구매자의 현황" << endl;
	buyer.ShowBuyResult();
}
