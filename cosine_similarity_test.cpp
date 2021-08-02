#include <iostream>
#include <time.h>
#include "NumCpp.hpp"
#include "cnpy.h"

using namespace nc;
using namespace std;
using namespace cnpy;


inline double _calCosineSimilarity(NdArray<double> vec1, NdArray<double> vec2)
{
	NdArray<double> zero_check_vector = vec1 * vec2;
	zero_check_vector = zero_check_vector[zero_check_vector > 0.];

	if (zero_check_vector.numCols() == 0)
	{
		return 0;
	}

	NdArray<double> vec1_for_dot = vec1[vec1 > 0.];
	NdArray<double> vec2_for_dot = vec2[vec2 > 0.];

	NdArray<double> vec1_for_norm = vec1[vec1 > 0.];
	NdArray<double> vec2_for_norm = vec2[vec2 > 0.];

	double dot_value = dot(vec1_for_dot, vec2_for_dot)(0, 0);
	double vec1_norm = norm(vec1_for_norm)(0, 0);
	double vec2_norm = norm(vec2_for_norm)(0, 0);


	double cosine_similarity = dot_value / (vec1_norm * vec2_norm);

	return cosine_similarity;
}



int main()
{
	NdArray<double> vec1 = { 0.1, 0.5, 0.4, 0.2, 0 };
	NdArray<double> vec2 = { 0.3, 0.7, 0, 0.5, 0.4 };


	NdArray<double> check_vector = vec1 * vec2;
	cout << check_vector << endl;

	cout << check_vector[check_vector > 0.] << endl;

	check_vector.shape().print();

	cout << check_vector.shape().cols << endl;
	

	NdArray<double> vec1_for_dot = vec1[vec1 > 0.];
	NdArray<double> vec2_for_dot = vec2[vec2 > 0.];

	vec1_for_dot.print();
	vec2_for_dot.print();

	cout << "-------------------------" << endl;

    /*int dotted = dot(vec1, vec2).item*/
	cout << vec1.numCols() << endl;
	cout << dot(vec1, vec2)(0,0) << endl;
	
	NdArray<double> temp = append(vec1, vec2, Axis::ROW);
	

	NdArray<double> t = fromfile<double>("c:/etc/code/array.txt", "\n");
	t = t.reshape(2, 29027);
	cout << t.shape();

	
	cout << t(0, t.cSlice());

	clock_t start = clock();
	for (int i = 0; i < 1000; i++) {
		_calCosineSimilarity(t(0, t.cSlice()), t(1, t.cSlice()));
	}	
	clock_t end = clock();
	cout << printf("Time: %lf\n", (double)(end - start)/CLOCKS_PER_SEC) << endl;
	

	/*NdArray<double> tfidf_array = load<double>("c:/etc/code/News/tfidf_array2.npy");
	cout<<tfidf_array.shape();*/
	return 0;
}