#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <time.h>

// Print matrix A
void print_matrix(const std::vector<std::vector<int> >& A_mat, int m, int n) {
    for (int k = 0; k < A_mat.size(); k++) {
        for (int l = 0; l < A_mat[k].size(); l++) {
            std::cout << A_mat[k][l] << " ";
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

    omp_set_num_threads(num_threads);

    // Read in matricies
    int n1;
    int m1;
    std::ifstream file;
    file.open(file1.c_str());
    
    file >> m1 >> n1;
    std::vector<std::vector<int> > A_mat(m1);

    int num;
    for (int i = 0; i < m1; i++) {
        std::vector<int> row(n1);
        for (int j = 0; j < n1; j++) {
            file >> num;
            row[j] = num;
        }
        A_mat[i] = row;
    }

    file.close();
    
    int n2;
    int m2;
    file.open(file2.c_str());
    file >> m2 >> n2;
    std::vector<std::vector<int> > B_mat(m2);

    for (int i = 0; i < m2; i++) {
        std::vector<int> row(n2);
        for (int j = 0; j < n2; j++) {
            file >> num;
            row[j] = num;
        }
        B_mat[i] = row;
    }

    file.close();

    std::vector<std::vector<int> > C_mat(m1);
    for (int i = 0; i < C_mat.size(); i++) {
        std::vector<int> row(n2);
        C_mat[i] = row;
    }
   
    double dtime = omp_get_wtime();
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < A_mat.size(); i++) {
            std::vector<int> row = A_mat[i];
            for (j = 0; j < n2; j++) {
                std::vector<int> col(m2);
                for (k = 0; k < m2; k++) {
                    col[k] = B_mat[k][j];
                }

                // Dot product
                int prod = 0;
                for (k = 0; k < row.size(); k++) {
                    prod += row[k]*col[k];
                }
                C_mat[i][j] = prod;
            }
        }
    }
    dtime = omp_get_wtime() - dtime;
    std::cout << "Time: " << dtime << "\n";

    std::cout << m1 << " " << n2 << "\n";
    print_matrix(C_mat, m1, n2);
}


