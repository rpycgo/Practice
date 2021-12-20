#include <iostream>
#include <queue>
#include <vector>

using namespace std;

void bfs(int edge);
int cnt = 0;
vector<vector<int>> network;
vector<bool> visited;



int main() {
	int node, edge, u, v;
	cin >> node >> edge;

	network.resize(node + 1);
	visited.resize(node + 1);
	
	for (int i = 0; i < edge; i++) {
		cin >> u >> v;
		network[u].push_back(v);
		network[v].push_back(u);
	}

	bfs(1);
	cout << cnt - 1;
}



void bfs(int edge) {
	int next_;
	queue<int> queue;
	queue.push(edge);

	while (!queue.empty()) {
		next_ = queue.front();
		queue.pop();
		for (int next : network[next_]) {
			if (!visited[next]) {
				visited[next] = true;
				queue.push(next);
				cnt++;
			}
		}
	}

}