#include <iostream>

using namespace std;


class Point
{
private:
	int xpos, ypos;

public:
	void Init(int x, int y);
	void ShowPositionInfo() const;
};

class Circle
{
private:
	int r;
	Point center;

public:
	void Init(int x, int y, int R);
	void ShowCircleInfo() const;
};

class Ring
{
private:
	Circle inCircle;
	Circle outCircle;

public:
	void Init(int inX, int inY, int inR, int outX, int outY, int outR);
	void ShowRingInfo() const;
};


void Point::Init(int x, int y)
{
	xpos = x;
	ypos = y;
}

void Point::ShowPositionInfo() const
{
	cout << "[" << xpos << ", " << ypos << "]" << endl;
}


void Circle::Init(int x, int y, int R) 
{
	r = R;
	center.Init(x, y);
}

void Circle::ShowCircleInfo() const
{
	cout << "radius: " << r << endl;
	center.ShowPositionInfo();
}


void Ring::Init(int inX, int inY, int inR, int outX, int outY, int outR)
{
	inCircle.Init(inX, inY, inR);
	outCircle.Init(outX, outY, outR);
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
	Ring ring;
	ring.Init(1, 1, 4, 2, 2, 9);
	ring.ShowRingInfo();
}
