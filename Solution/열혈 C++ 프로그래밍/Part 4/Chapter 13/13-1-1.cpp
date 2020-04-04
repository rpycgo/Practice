#include <iostream>

using namespace std;


class Point
{
private:
	int xpos, ypos;

public:
	Point(int x = 0, int y = 0) : xpos(x), ypos(y) 
	{}

	void ShowPosition() const;
};


void Point::ShowPosition() const
{
	cout << "[" << xpos << ", " << ypos << "]" << endl;
}


template <typename T>
void SwapData(T& coord1, T& coord2)
{
	T temp = coord1;
	coord1 = coord2;
	coord2 = temp;
}


int main()
{
	Point coord1(1, 2);
	Point coord2(10, 20);

	SwapData(coord1, coord2);

	coord1.ShowPosition();
	coord2.ShowPosition();
}
