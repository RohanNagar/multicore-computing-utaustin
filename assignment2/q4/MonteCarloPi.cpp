#define RADIUS 1

#include <cmath>
#include <random>
#include <string>
#include <iostream>
#include <limits>
#include <ctime>
#include <omp.h>

typedef std::numeric_limits< double > dbl;

double MonteCarloPi(int s);
bool inCircle(double x, double y, double radius);
double randInRange(double lower, double upper);

using namespace std;

int main(int argc, char** argv){
	if(argc < 2)
	{
		exit(-1);
	}
	int s = std::stoi(argv[1]);
	double pi = MonteCarloPi(s);
	cout.precision(dbl::max_digits10);
	cout << "Estimate of pi: " << fixed << pi << endl;
}

double randInRange(double lower, double upper){
	if(lower > upper){
		double temp = lower;
		lower = upper;
		upper = temp;	
	}

	std::uniform_real_distribution<double> unif(lower, upper);
	//std:: default_random_engine re;
	std::mt19937 rng;
	rng.seed(std::random_device{}());
	double a_random_double = unif(rng);
	return a_random_double;
}

bool inCircle(double x, double y, double radius) {
	return (pow(x, 2) + pow(y, 2)) < pow(radius, 2);
}

double MonteCarloPi(int s) {
	double c = 0;
	int i;
	double pi;
	int x_counter = 0;
	int y_counter = 0;

	double lower = 0;
	double upper = 2*RADIUS;
	#pragma omp parallel for reduction(+:c)
	{
		std::uniform_real_distribution<double> unif(lower, upper);
		//std:: default_random_engine rng;
		std::mt19937 rng;
		rng.seed(std::time(0));
	
	
		for(i = 0; i < s; i++)
		{
			//choose two random numbers in range 0 to 2RADIUS
			double x = unif(rng);
			double y = unif(rng);
			cout << "random x, y: " << x << ", " << y << endl;
			if(x < RADIUS)
			{
				x_counter++;
			}
			if(y < RADIUS)
			{
				y_counter++;
			}
			//check if in circle
			bool ic = inCircle(x, y, 2*RADIUS);
	 		cout << "inCircle: " << inCircle << endl;
		
			if(ic)
			{
				c += 1;
			}
		}
	}
	pi = c/s*4;
 	cout << x_counter << " out of " << s << " were less than 0.5" << endl;
	cout << y_counter << " out of " << s << " were less than 0.5" << endl;
	cout << c << " points in circle out of " << s << endl;
	return pi;
}
