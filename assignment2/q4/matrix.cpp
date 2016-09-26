#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>

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


}


