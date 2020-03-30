#include <iostream>

using namespace std;


class Calculator
{
private:
	float x, y;
	int AddCnt, MinCnt, MultiCnt, DivCnt;
public:
	float Add(float x, float y);	
	float Min(float x, float y);
	float Multi(float x, float y);
	float Div(float x, float y);

	void Init();
	void ShowOpCount();
};


float Calculator::Add(float x, float y)
{
	AddCnt++;

	return x + y;
}

float Calculator::Min(float x, float y)
{
	MinCnt++;

	return x - y;
}

float Calculator::Multi(float x, float y)
{
	MultiCnt++;

	return x * y;
}

float Calculator::Div(float x, float y)
{
	DivCnt++;

	return x / y;
}

void Calculator::Init()
{
	AddCnt = MinCnt = MultiCnt = DivCnt = 0;
}

void Calculator::ShowOpCount()
{
	cout << "+: " << AddCnt << " ";
	cout << "-: " << MinCnt << " ";
	cout << "*: " << MultiCnt << " ";
	cout << "/: " << DivCnt << endl;
}


int main(void)
{
	Calculator cal;
	cal.Init();
	cout << "3.2 + 2.4 = " << cal.Add(3.2, 2.4) << endl;
	cout << "3.5 / 1.7 = " << cal.Div(3.2, 2.4) << endl;
	cout << "2.2 - 1.5 = " << cal.Min(3.2, 2.4) << endl;
	cout << "4.9 / 1.2 = " << cal.Div(3.2, 2.4) << endl;
	cal.ShowOpCount();

	return 0;
}