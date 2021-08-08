/* O(NlogN) */

#include <iostream>
#include <vector>
#include <queue>

using namespace std;

int number = 6;
int INF = 1000000;

vector<pair<int, int>> distanceMatrix[7];
int shortestDistance[7];


void dijkstra(int start)
{
	shortestDistance[start] = 0;
	priority_queue<pair<int, int>> pq;
	pq.push(make_pair(start, 0));

	while (!pq.empty())
	{
		int current = pq.top().first;
		int distance = -pq.top().second;
		pq.pop();
		
		if (shortestDistance[current] < distance)
		{
			continue;
		}

		for (int i = 0; i < distanceMatrix[current].size(); i++)
		{
			int next = distanceMatrix[current][i].first;
			int nextDistance = distance + distanceMatrix[current][i].second;

			if (nextDistance < shortestDistance[next])
			{
				shortestDistance[next] = nextDistance;
				pq.push(make_pair(next, -nextDistance));
			} 
		}
	}
}


int main(void)
{
	for (int i = 1; i <= number; i++)
	{
		shortestDistance[i] = INF;
	}

	distanceMatrix[1].push_back(make_pair(2, 2));
	distanceMatrix[1].push_back(make_pair(3, 5));
	distanceMatrix[1].push_back(make_pair(4, 1));

	distanceMatrix[2].push_back(make_pair(1, 2));
	distanceMatrix[2].push_back(make_pair(3, 3));
	distanceMatrix[2].push_back(make_pair(4, 2));

	distanceMatrix[3].push_back(make_pair(1, 5));
	distanceMatrix[3].push_back(make_pair(2, 3));
	distanceMatrix[3].push_back(make_pair(4, 3));
	distanceMatrix[3].push_back(make_pair(5, 1));
	distanceMatrix[3].push_back(make_pair(6, 5));
	
	distanceMatrix[4].push_back(make_pair(1, 1));
	distanceMatrix[4].push_back(make_pair(2, 2));
	distanceMatrix[4].push_back(make_pair(3, 3));
	distanceMatrix[4].push_back(make_pair(5, 1));

	distanceMatrix[5].push_back(make_pair(3, 1));
	distanceMatrix[5].push_back(make_pair(4, 1));
	distanceMatrix[5].push_back(make_pair(6, 2));

	distanceMatrix[6].push_back(make_pair(3, 5));
	distanceMatrix[6].push_back(make_pair(5, 2));

	dijkstra(1);

	for (int i = 1; i <= number; i++)
	{
		printf("%d ", shortestDistance[i]);
	}
}
