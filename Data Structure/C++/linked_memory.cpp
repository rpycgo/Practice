#include <iostream>

using namespace std;


template <typename T>
class List {
public:
	const T& operator[](unsigned index);
	void insertAtFront(const T& data);

private:
	class ListNode {
	public:
		const T& data;
		ListNode* next;
		ListNode(const T& data) : data(data), next(NULL) { }
	};

	ListNode* head_; /* <Head pointer for our List> */
};


template <typename T>
const T& List<T>::operator[](unsigned index) {
	// Start a 'thru' pointer to advance thru the list:
	ListNode* thru = head_;

	// Loop until the end of the list (or until a 'nullptr'):
	while (index > 0 && thru->next != nullptr) {
		thru = thru->next;
		index--;
	}

	// Return the data:
	return thru->data;
}


template <typename T>
void List<T>::insertAtFront(const T& data) {
	// Create a new ListNode on the heap:
	ListNode* node = new ListNode(data);

	// Set the new node's next pointer point the current head of the List:
	node->next = head_;

	// Set the List's head pointer to be the new noode:
	head_ = node;
}




int main() {
	List<int> list;

	cout << "Inserting element 3 at front..." << endl;
	list.insertAtFront(3);
	cout << "list[0]: " << list[0] << endl;

	cout << "Inserting element 30 at front..." << endl;
	list.insertAtFront(30);
	cout << "list[0]: " << list[0] << endl;
	cout << "list[1]: " << list[1] << endl;
}
