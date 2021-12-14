#include "graphics.h"
#include "omp.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <memory>

namespace {

void print_info(int ms, int threads_count) {
    if (!threads_count) {
        threads_count = omp_get_max_threads();
    }
    std::cout << "Time (" << std::to_string(threads_count) << " thread";
    if (threads_count > 1) {
        std::cout << 's';
    }
    std::cout << "): " << std::to_string(ms) << " ms" << std::endl;
}

}

const int NUMBER_OF_DEFAULT_ARGUMENTS = 4;

int main(int argc, char * argv[]) {
    try {
        if (argc < NUMBER_OF_DEFAULT_ARGUMENTS) {
            throw std::invalid_argument("wrong number of arguments.");
        }
        int threads_count = atoi(argv[1]);
        std::string input_file_name = std::string(argv[2]),
        output_file_name = std::string(argv[3]);
        double ignore_coefficient = 0;
        if (argc >= NUMBER_OF_DEFAULT_ARGUMENTS + 1) {
            ignore_coefficient = atof(argv[4]);
        }

        std::ifstream in(input_file_name, std::ios::binary);
        in.exceptions(std::ifstream::failbit | std::ifstream::eofbit);
        auto image = std::unique_ptr<Graphics::Image>(Graphics::Image::read_image(in));

        auto start_time = omp_get_wtime();
        image->normalize(threads_count, ignore_coefficient);
        auto end_time = omp_get_wtime();
        int ms = static_cast<int>((end_time - start_time) * 1000);

        print_info(ms, threads_count);
        std::ofstream out(output_file_name, std::ios::binary);
        out << *image;
    } catch (const std::invalid_argument& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::ios_base::failure& e) {
        std::cout << "Failed to read input file: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
