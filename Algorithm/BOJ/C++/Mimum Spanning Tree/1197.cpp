#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int getParent(int* parent, int x) {
	if (parent[x] == x) {
		return x;
	}
	else
	{ 
		return parent[x] = getParent(parent, parent[x]);
	}
	
}

void unionParent(int parent[], int a, int b) {
	a = getParent(parent, a);
	b = getParent(parent, b);
	
	if (a < b) {
		parent[b] = a;
	}
	else {
		parent[a] = b;
	}
}

int findParent(int parent[], int a, int b) {
	a = getParent(parent, a);
	b = getParent(parent, b);

	if (a == b) {
		return 1;
	}
	else {
		return 0;
	}
}

class Node {
public:
	int node[2];
	int distance;
	Node(int a, int b, int distance) {
		this->node[0] = a;
		this->node[1] = b;
		this->distance = distance;
	}

	bool operator <(Node& node) {
		return this->distance < node.distance;
	}
};

int main() {
	int V, E;
	cin >> V >> E;
	
	vector<Node> nodes;

	for (int i = 0; i < E; i++) {
		int A, B, C;
		cin >> A >> B >> C;
		nodes.push_back(Node(A, B, C));
	}

	sort(nodes.begin(), nodes.end());
	
	int* parent = new int[V];
	for (int i = 0; i < V; i++) {
		parent[i] = i;
	}

	int sum = 0;
	for (unsigned int i = 0; i < nodes.size(); i++) {
		if (!findParent(parent, nodes[i].node[0] - 1, nodes[i].node[1] - 1)) {
			sum += nodes[i].distance;
			unionParent(parent, nodes[i].node[0] - 1, nodes[i].node[1] - 1);
		}
	}

	cout << sum << endl;

	delete[] parent;
}
