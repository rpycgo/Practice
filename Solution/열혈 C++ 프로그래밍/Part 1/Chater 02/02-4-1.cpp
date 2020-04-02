#include <iostream>
#include <cstring>

using namespace std;


void main()
{
	const char *seq1 = "Hello World!";
	char seq2[30];

	cout << seq1 << endl;
	cout << strlen(seq1) << endl;
	strcpy_s(seq2, seq1);
	cout << seq2 << endl;
	strcat_s(seq2, seq1);
	cout << seq2 << endl;

	if (strcmp(seq1, seq2) == 0)
		cout << "T" << endl;
	else
		cout << "F" << endl;
}
