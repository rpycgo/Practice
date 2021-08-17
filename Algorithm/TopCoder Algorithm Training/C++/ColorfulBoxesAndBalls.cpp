#include <algorithm>
#include <limits.h>
#include <iostream>

using namespace std;

class ColorfulBoxesAndBalls
{
private:
	int numRed, numBlue, onlyRed, onlyBlue, bothColors;
	const int MIN_VALUE = -1000000;

public:
	int answer;
	ColorfulBoxesAndBalls(int numRed, int numBlue, int onlyRed, int onlyBlue, int bothColors);
	void getMaximum();
};


ColorfulBoxesAndBalls::ColorfulBoxesAndBalls(int numRed, int numBlue, int onlyRed, int onlyBlue, int bothColors)
{
	this->numRed = numRed;
	this->numBlue = numBlue;
	this->onlyRed = onlyRed;
	this->onlyBlue = onlyBlue;
	this->bothColors = bothColors;

	ColorfulBoxesAndBalls::getMaximum();
}

void ColorfulBoxesAndBalls::getMaximum()
{
	int answer = this->MIN_VALUE;
	int change = min(numRed, numBlue);

	for (int i = 0; i <= change; i++) 
	{
		int currentScore = (numRed - i) * onlyRed + (numBlue - i) * onlyBlue + 2 * i * bothColors;

		answer = max(answer, currentScore);
	}
	
	this->answer = answer;
}


int main()
{
	ColorfulBoxesAndBalls colorfulboxesandballs(2, 3, 100 ,400 ,200);

	cout << colorfulboxesandballs.answer << endl;
}
