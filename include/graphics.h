#ifndef HW5_GRAPHICS_H
#define HW5_GRAPHICS_H

#include <iosfwd>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace Graphics {

class RGB_pixel {
public:
    RGB_pixel() = default;

    explicit RGB_pixel(std::uint8_t r, std::uint8_t g, std::uint8_t b);

    [[nodiscard]] std::uint8_t get_red() const noexcept {
        return red;
    }

    [[nodiscard]] std::uint8_t get_green() const noexcept {
        return green;
    }

    [[nodiscard]] std::uint8_t get_blue() const noexcept {
        return blue;
    }

    void set_red(std::uint8_t value) {
        red = value;
    }

    void set_green(std::uint8_t value) {
        green = value;
    }

    void set_blue(std::uint8_t value) {
        blue = value;
    }

    friend std::ifstream& operator >>(std::ifstream& in, RGB_pixel& pixel);
    friend std::ofstream& operator <<(std::ofstream& out, const RGB_pixel& pixel);

private:
    std::uint8_t red, green, blue;
};

class grayscale_pixel {
public:
    grayscale_pixel() = default;

    explicit grayscale_pixel(std::uint8_t l);

    [[nodiscard]] std::uint8_t get_luminance() const noexcept {
        return luminance;
    }

    void set_luminance(std::uint8_t new_luminance) {
        luminance = new_luminance;
    }

    friend std::ifstream& operator >>(std::ifstream& in, grayscale_pixel& pixel);
    friend std::ofstream& operator <<(std::ofstream& out, const grayscale_pixel& pixel);

private:
    std::uint8_t luminance;
};

static const std::size_t MAX_VALUE = 255;

class Image {
public:
    Image() = default;
    ~Image() = default;

    virtual void normalize(int threads_count, double ignore_coefficient) = 0;

    friend std::ifstream& operator >>(std::ifstream& in, Image& image);
    friend std::ofstream& operator <<(std::ofstream& out, const Image& image);

    static Image* read_image(std::ifstream& in);

protected:
    virtual std::ofstream& print(std::ofstream& out) const;
    virtual std::ifstream& read(std::ifstream& in);

    std::size_t width = 0, height = 0;

    static const std::string PGM_FORMAT;
    static const std::string PPM_FORMAT;

private:
    static Image* create_image_by_type(const std::string& type);
};

typedef std::vector<std::vector<grayscale_pixel>> grayscale_matrix;
class ImagePGM : public Image {
public:
    ImagePGM() = default;
    ~ImagePGM() = default;

    void normalize(int threads_count, double ignore_coefficient) override;

private:
    std::ofstream& print(std::ofstream& out) const override;
    std::ifstream& read(std::ifstream& in) override;

    grayscale_matrix matrix;
};

typedef std::vector<std::vector<RGB_pixel>> RGB_matrix;
class ImagePPM : public Image {
public:
    ImagePPM() = default;
    ~ImagePPM() = default;

    void normalize(int threads_count, double ignore_coefficient) override;

private:
    std::ofstream& print(std::ofstream& out) const override;
    std::ifstream& read(std::ifstream& in) override;

    RGB_matrix matrix;
};

}
#endif
