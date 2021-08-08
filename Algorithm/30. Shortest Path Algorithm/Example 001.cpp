/* O(n^2) */

#include <iostream>

using namespace std;

int number = 6;
int INF = 1000000;
bool visitedNode[6];
int shortestDistance[6];

int distanceMatrix[6][6] = {
	{0, 2, 5, 1, INF, INF},
	{2, 0, 3, 2, INF, INF},
	{5, 3, 0, 3, 1, 5},
	{1, 2, 3, 0, 1, INF},
	{INF, INF, 1, 1, 0, 2},
	{INF, INF, 5, INF, 2, 0},
};

int getSmallIndex() 
{
	int min = INF;
	int index = 0;
	for (int i = 0; i < number; i++)
	{
		if (shortestDistance[i] < min && !visitedNode[i])
		{
			min = shortestDistance[i];
			index = i;
		}
	}
	return index;
}

void dijkstra(int start) 
{
	for (int i = 0; i < number; i++)
	{
		shortestDistance[i] = distanceMatrix[start][i];
	}
	
	visitedNode[start] = true;

	for (int i = 0; i < number - 2; i++) 
	{
		int current = getSmallIndex();
		visitedNode[current] = true;

		for (int j = 0; j < 6; j++)
		{
			if (!visitedNode[j])
			{
				if (shortestDistance[current] + distanceMatrix[current][j] < shortestDistance[j])
				{
					shortestDistance[j] = shortestDistance[current] + distanceMatrix[current][j];
				}
			}
		}
	}
}


int main(void)
{
	dijkstra(0);

	for (int i = 0; i < number; i++)
	{
		cout << shortestDistance[i] ;
	}
}
