#include <iostream>

using namespace std;


class Point
{
private:
	int xpos, ypos;

public:
	Point(int x, int y) : xpos(x), ypos(y) {}
	void ShowPositionInfo() const;
};

class Circle
{
private:
	int r;
	Point center;

public:
	Circle(int x, int y, int R) : center(x, y)
	{
		r = R;
	}
	void ShowCircleInfo() const;
};

class Ring
{
private:
	Circle inCircle;
	Circle outCircle;

public:
	Ring(int inX, int inY, int inR, int outX, int outY, int outR)
		: inCircle(inX, inY, inR), outCircle(outX, outY, outR) {}
	void ShowRingInfo() const;
};



void Point::ShowPositionInfo() const
{
	cout << "[" << xpos << ", " << ypos << "]" << endl;
}


void Circle::ShowCircleInfo() const
{
	cout << "radius: " << r << endl;
	center.ShowPositionInfo();
}


void Ring::ShowRingInfo() const
{
	cout << "Inner Circle Info..." << endl;
	inCircle.ShowCircleInfo();	
	cout << "Outer Circle Info..." << endl;
	outCircle.ShowCircleInfo();
}


int main()
{
	Ring ring(1, 1, 4, 2, 2, 9);	
	ring.ShowRingInfo();
}
