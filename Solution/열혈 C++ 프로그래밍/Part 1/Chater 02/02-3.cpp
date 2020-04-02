#include <iostream>
#include <cstdlib>

using namespace std;


typedef struct __Point
{
	int xpos, ypos;
} Point;


Point &PntAdder(const Point &p1, const Point &p2)
{
	Point* coord = new Point;
	coord->xpos = p1.xpos + p2.xpos;
	coord->ypos = p1.ypos + p2.ypos;

	return *coord;
}



void main()
{
	Point* coord1 = new Point;
	coord1->xpos = 1;
	coord1->ypos = 2;

	Point* coord2 = new Point;
	coord2->xpos = 3;
	coord2->ypos = 4;

	Point& coordinate = PntAdder(*coord1, *coord2);

	cout << "[" << coordinate.xpos << ", " << coordinate.ypos << "]" << endl;

	delete coord1, coord2, & coordinate;
}
