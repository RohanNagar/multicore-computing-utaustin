#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <omp.h>

void read_matrix(std::string filename, double* A, int m, int n) {

}

// Print matrix A
void print_matrix(double* A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i*m+j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    if (argc <= 3) {
        std::cout << "Usage: " << argv[0] << " <file1> <file2> <num_threads>\n";
        exit(1);
    }

    // Get the command line args
    std::string file1 = argv[1];
    std::string file2 = argv[2];

    int num_threads;
    std::stringstream convert(argv[3]);

    // Convert the num_threads to an integer.
    // If fail to convert, print an error and continue with one thread.
    if (!(convert >> num_threads)) {
        std::cout << "Warning: The number of threads specified is not a numeric value.\n";
        std::cout << "Continuing execution with one thread.\n";
        num_threads = 1;
    }

    // Read in matricies
    //read_matrix(file1, ...);
    //read_matrix(file2, ...);

    // Samples
    //int n = 3;
    //int m = 3;
    //double* A = (double*) malloc(sizeof(double)*m*n);
    //double* A2 = (double*) malloc(sizeof(double)*m*n);
    //for(int i = 0; i < m*n; i++) {
    //    A[i] = 1.0*rand()/RAND_MAX;
    // }

}


