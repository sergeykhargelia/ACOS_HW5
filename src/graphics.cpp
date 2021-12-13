#include "graphics.h"
#include "omp.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace Graphics {

RGB_pixel::RGB_pixel(std::uint8_t r, std::uint8_t g, std::uint8_t b) : red(r), green(g), blue(b) {}

std::ifstream& operator >>(std::ifstream& in, RGB_pixel& pixel) {
    in.read(reinterpret_cast<char *>(&pixel.red), sizeof(pixel.red));
    in.read(reinterpret_cast<char *>(&pixel.green), sizeof(pixel.green));
    in.read(reinterpret_cast<char *>(&pixel.blue), sizeof(pixel.blue));
    return in;
}

std::ofstream& operator <<(std::ofstream& out, const RGB_pixel& pixel) {
    out.write(reinterpret_cast<const char *>(&pixel.red), sizeof(pixel.red));
    out.write(reinterpret_cast<const char *>(&pixel.green), sizeof(pixel.green));
    out.write(reinterpret_cast<const char *>(&pixel.blue), sizeof(pixel.blue));
    return out;
}

grayscale_pixel::grayscale_pixel(std::uint8_t l) : luminance(l) {}

std::ifstream& operator >>(std::ifstream& in, grayscale_pixel& pixel) {
    in.read(reinterpret_cast<char *>(&pixel.luminance), sizeof(pixel.luminance));
    return in;
}

std::ofstream& operator <<(std::ofstream& out, const grayscale_pixel& pixel) {
    out.write(reinterpret_cast<const char *>(&pixel.luminance), sizeof(pixel.luminance));
    return out;
}

std::ofstream& Image::print(std::ofstream& out) const {
    std::string info = std::to_string(width) + " " + std::to_string(height) + "\n" + std::to_string(MAX_VALUE) + "\n";
    out.write(info.c_str(), static_cast<long long>(info.size()));
    return out;
}

std::ofstream& operator <<(std::ofstream& out, const Image& image) {
    return image.print(out);
}

std::ifstream& Image::read(std::ifstream& in) {
    std::string buf;
    std::getline(in, buf);
    auto space_position = buf.find(' ');
    width = std::stoi(std::string(buf.begin(), buf.begin() + space_position));
    height = std::stoi(std::string(buf.begin() + space_position + 1, buf.end()));
    std::getline(in, buf);
    if (std::stoi(buf) != MAX_VALUE) {
        throw std::invalid_argument("Wrong format of the input file -- maximum possible value for pixel is not 255.");
    }
    return in;
}

std::ifstream& operator >>(std::ifstream& in, Image& image) {
    return image.read(in);
}

const std::string Image::PGM_FORMAT = "P5";
const std::string Image::PPM_FORMAT = "P6";

Image* Image::create_image_by_type(const std::string& type) {
    if (type == PGM_FORMAT) {
        return new ImagePGM();
    } else if (type == PPM_FORMAT) {
        return new ImagePPM();
    } else {
        throw std::invalid_argument("Unknown format of the image.");
    }
}

Image* Image::read_image(std::ifstream& in) {
    std::string type;
    std::getline(in, type);
    auto image = create_image_by_type(type);
    in >> *image;
    return image;
}

std::ifstream& ImagePGM::read(std::ifstream& in) {
    Image::read(in);
    matrix.resize(height, std::vector<grayscale_pixel>(width));
    for (std::size_t row = 0; row < height; row++) {
        for (std::size_t column = 0; column < width; column++) {
            in >> matrix[row][column];
        }
    }
    return in;
}

std::ofstream& ImagePGM::print(std::ofstream& out) const {
    out.write(PGM_FORMAT.c_str(), static_cast<long long>(PGM_FORMAT.size()));
    out.write("\n", sizeof(char));
    Image::print(out);
    for (std::size_t row = 0; row < height; row++) {
        for (std::size_t column = 0; column < width; column++) {
            out << matrix[row][column];
        }
    }
    return out;
}

static std::uint8_t calculate_min_luminance(const std::array<int, MAX_VALUE + 1>& cnt, int ignore_count) {
    int prefix_sum = 0;
    for (std::size_t i = 0; i <= MAX_VALUE; i++) {
        prefix_sum += cnt[i];
        if (prefix_sum > ignore_count) {
            return static_cast<std::uint8_t>(i);
        }
    }
    return MAX_VALUE;
}

static std::uint8_t calculate_max_luminance(const std::array<int, MAX_VALUE + 1>& cnt, int ignore_count) {
    int suffix_sum = 0;
    for (std::size_t i = MAX_VALUE; i >= 0; i--) {
        suffix_sum += cnt[i];
        if (suffix_sum > ignore_count) {
            return static_cast<std::uint8_t>(i);
        }
    }
    return 0;
}

static inline std::uint8_t calculate_new_value (
        std::uint8_t old_value,
        std::uint8_t diff,
        std::uint8_t min_value,
        std::uint8_t max_value
) {
    if (old_value < min_value) {
        return 0;
    }
    if (old_value > max_value) {
        return MAX_VALUE;
    }
    if (diff > 0) {
        return (old_value - min_value) * MAX_VALUE / diff;
    }
    return old_value;
}

void ImagePGM::normalize(int threads_count, double ignore_coefficient) {
    if (threads_count) {
        omp_set_num_threads(threads_count);
    }

    std::array<int, MAX_VALUE + 1> cnt;
    std::fill(cnt.begin(), cnt.end(), 0);

    #pragma omp parallel default(none) shared(cnt, width, height, matrix)
    {
        std::array<int, MAX_VALUE + 1> local_cnt;
        std::fill(local_cnt.begin(), local_cnt.end(), 0);

        #pragma omp for
        for (std::size_t row = 0; row < height; row++) {
            for (std::size_t column = 0; column < width; column++) {
                local_cnt[matrix[row][column].get_luminance()]++;
            }
        }

        #pragma omp critical
        {
            for (std::size_t value = 0; value <= MAX_VALUE; value++) {
                cnt[value] += local_cnt[value];
            }
        }
    }

    int ignore_count = (int)(ignore_coefficient * (double)width * (double)height);

    std::uint8_t min_luminance = calculate_min_luminance(cnt, ignore_count),
    max_luminance = calculate_max_luminance(cnt, ignore_count);

    std::uint8_t diff = max_luminance - min_luminance;

    #pragma omp parallel for default(none) shared(min_luminance, max_luminance, diff, height, width, matrix)
    for (std::size_t row = 0; row < height; row++) {
        for (std::size_t column = 0; column < width; column++) {
            matrix[row][column].set_luminance(calculate_new_value(
                    matrix[row][column].get_luminance(), diff, min_luminance, max_luminance
            ));
        }
    }
}

std::ifstream& ImagePPM::read(std::ifstream& in) {
    Image::read(in);
    matrix.resize(height, std::vector<RGB_pixel>(width));
    for (std::size_t row = 0; row < height; row++) {
        for (std::size_t column = 0; column < width; column++) {
            in >> matrix[row][column];
        }
    }
    return in;
}

std::ofstream& ImagePPM::print(std::ofstream& out) const {
    out.write(PPM_FORMAT.c_str(), static_cast<long long>(PPM_FORMAT.size()));
    out.write("\n", sizeof(char));
    Image::print(out);
    for (std::size_t row = 0; row < height; row++) {
        for (std::size_t column = 0; column < width; column++) {
            out << matrix[row][column];
        }
    }
    return out;
}

void ImagePPM::normalize(int threads_count, double ignore_coefficient) {
    if (threads_count) {
        omp_set_num_threads(threads_count);
    }

    std::array<int, MAX_VALUE + 1> cnt_red, cnt_green, cnt_blue;
    std::fill(cnt_red.begin(), cnt_red.end(), 0);
    std::fill(cnt_green.begin(), cnt_green.end(), 0);
    std::fill(cnt_blue.begin(), cnt_blue.end(), 0);

    #pragma omp parallel default(none) shared(cnt_red, cnt_green, cnt_blue, width, height, matrix)
    {
        std::array<int, MAX_VALUE + 1> local_cnt_red, local_cnt_green, local_cnt_blue;
        std::fill(local_cnt_red.begin(), local_cnt_red.end(), 0);
        std::fill(local_cnt_green.begin(), local_cnt_green.end(), 0);
        std::fill(local_cnt_blue.begin(), local_cnt_blue.end(), 0);

        #pragma omp for
        for (std::size_t row = 0; row < height; row++) {
            for (std::size_t column = 0; column < width; column++) {
                local_cnt_red[matrix[row][column].get_red()]++;
                local_cnt_green[matrix[row][column].get_green()]++;
                local_cnt_blue[matrix[row][column].get_blue()]++;
            }
        }

        #pragma omp critical
        {
            for (std::size_t value = 0; value <= MAX_VALUE; value++) {
                cnt_red[value] += local_cnt_red[value];
                cnt_green[value] += local_cnt_green[value];
                cnt_blue[value] += local_cnt_blue[value];
            }
        }
    }
    int ignore_count = (int)(ignore_coefficient * (double)width * (double)height);

    std::uint8_t min_luminance = std::min({
        calculate_min_luminance(cnt_red, ignore_count),
        calculate_min_luminance(cnt_green, ignore_count),
        calculate_min_luminance(cnt_blue, ignore_count)}),

    max_luminance = std::max({
         calculate_max_luminance(cnt_red, ignore_count),
         calculate_max_luminance(cnt_green, ignore_count),
         calculate_max_luminance(cnt_blue, ignore_count)});

    std::uint8_t diff = max_luminance - min_luminance;

    #pragma omp parallel for default(none) shared(min_luminance, max_luminance, diff, height, width, matrix)
    for (std::size_t row = 0; row < height; row++) {
        for (std::size_t column = 0; column < width; column++) {
            matrix[row][column].set_red(calculate_new_value(
                    matrix[row][column].get_red(), diff, min_luminance, max_luminance
            ));

            matrix[row][column].set_green(calculate_new_value(
                    matrix[row][column].get_green(), diff, min_luminance, max_luminance
            ));

            matrix[row][column].set_blue(calculate_new_value(
                    matrix[row][column].get_blue(), diff, min_luminance, max_luminance
            ));
        }
    }
}

}