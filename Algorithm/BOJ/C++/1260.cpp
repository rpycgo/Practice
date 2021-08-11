#include <iostream>
#include <queue>
#include <vector>

using namespace std;


class Algorithm
{	
private:
	int vertex;
	int edge;
	int* bfs_check_vector;
	int* dfs_check_vector;
	vector<int> a[1001];

public:
	Algorithm(int vertex, int edge);
	void connectEdge();
	void BFS(int start);
	void DFS(int start);
};


Algorithm::Algorithm(int vertex, int edge) {

	this->vertex = vertex;
	this->edge = edge;

	this->bfs_check_vector = new int[(this->vertex + 1)]{};
	this->dfs_check_vector = new int[(this->vertex + 1)]{};

	Algorithm:: connectEdge();
}


void Algorithm::connectEdge() {
	
	int start_vertex, end_vertex;
	for (int i = 0; i < this->edge; i++) {
		cin >> start_vertex >> end_vertex;
		this->a[start_vertex].push_back(end_vertex);
		this->a[end_vertex].push_back(start_vertex);
	}

}


void Algorithm::BFS(int start) {
	
	queue<int> q;
	
	q.push(start);
	this->bfs_check_vector[start] = true;
	while (!q.empty()) {
		int x = q.front();
		q.pop();
		printf("%d ", x);		
		for (unsigned int i = 0; i < this->a[x].size(); i++) {
			int y = this->a[x][i];
			if (!this->bfs_check_vector[y]) {
				q.push(y);
				this->bfs_check_vector[y] = true;
			}
		}
	}

	delete[] this->bfs_check_vector;
}

void Algorithm::DFS(int start) {

	if (this -> dfs_check_vector[start]) {
		return;
	}
	
	this->dfs_check_vector[start] = true;
	cout << start << ' ';
	for (unsigned int i = 0; i < this->a[start].size(); i++) {
		DFS(this->a[start][i]);
	}

	
}




int main() {

	int vertex, edge, start;

	cin >> vertex >> edge >> start;
	
	Algorithm algorithm(vertex, edge);

	algorithm.DFS(start);
	cout << endl;
	algorithm.BFS(start);
	
	return 0;
}
