#include <iostream>
#include <queue>
#include <vector>

using namespace std;

int number = 7;
int c[7];
vector<int> a[8];


void bfs(int start) {
	queue<int> q;
	q.push(start);
	c[start] = true;
	while (!q.empty()) {
		int x = q.front();
		q.pop();
		printf("%d ", x);
			for (int i = 0; i < a[x].size(); i++) {
				int y = a[x][i];
				if (!c[y]) {
					q.push(y);
					c[y] = true;
				}
			}
	}
}





int main(void) {
	a[1].push_back(2);
	a[1].push_back(3);
	a[1].push_back(4);

	a[2].push_back(4);

	a[3].push_back(4);
	
	bfs(1);

	return 0;
}
