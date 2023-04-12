#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <Eigen/Core>

#include <curl/curl.h>
#include <zlib.h>

const std::string BASE_URL = "http://yann.lecun.com/exdb/mnist/";

/**
 * @brief Check if file exists
 * @param filename Name of file to check
 * @return True if file exists, false otherwise
 */
bool file_exists(const std::string &filename)
{
    std::ifstream infile(filename);
    return infile.good();
}

static size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
    size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
    return written;
}

/**
 * @brief Download file from URL and save to local path
 * @param url URL of file to download
 * @param path Local path to save downloaded file
 */
void download_file(const std::string &url, const std::string &path)
{
    CURL *curl_handle;
    FILE *out;

    curl_global_init(CURL_GLOBAL_ALL);

    /* init the curl session */
    curl_handle = curl_easy_init();

    /* set URL to get here */
    curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());

    /* Switch on full protocol/debug output while testing */
    curl_easy_setopt(curl_handle, CURLOPT_VERBOSE, 1L);

    /* disable progress meter, set to 0L to enable and disable debug output */
    curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 1L);

    /* send all data to this function  */
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_data);

    /* open the file */
    out = fopen(path.c_str(), "wb");
    if (out)
    {

        /* write the page body to this file handle */
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, out);

        /* get it! */
        curl_easy_perform(curl_handle);

        /* close the header file */
        fclose(out);
    }

    /* cleanup curl stuff */
    curl_easy_cleanup(curl_handle);

    curl_global_cleanup();
}

/**
 * @brief Unzip gz file and save to local path
 * @param gz_path Local path of gz file to unzip
 * @param out_path Local path to save unzipped file
 * @return True if unzipping was successful, false otherwise
 */
bool unzip_file(const std::string &gz_path, const std::string &out_path)
{
    gzFile gz = gzopen(gz_path.c_str(), "rb");
    if (!gz)
    {
        std::cerr << "Error opening gz file: " << gz_path << std::endl;
        return false;
    }

    std::ofstream outfile(out_path, std::ofstream::binary);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening output file: " << out_path << std::endl;
        return false;
    }

    char buffer[1024];
    int uncompressed_bytes;
    while ((uncompressed_bytes = gzread(gz, buffer, sizeof(buffer))) > 0)
    {
        outfile.write(buffer, uncompressed_bytes);
    }

    outfile.close();
    gzclose(gz);

    return true;
}

/**
 * @brief Read a MNIST image file
 *
 * @param filename Name of the file to read
 * @return A matrix containing the images
 */
Eigen::MatrixXd read_mnist_images(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cout << "Error: Failed to open file: " << filename.c_str() << std::endl;
        return Eigen::MatrixXd();
    }

    int magic_number = 0;
    int num_images = 0;
    int num_rows = 0;
    int num_cols = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);

    if (magic_number != 2051)
    {
        std::cout << "Error: Invalid magic number in file: " << filename.c_str() << std::endl;
        return Eigen::MatrixXd();
    }

    file.read((char *)&num_images, sizeof(num_images));
    file.read((char *)&num_rows, sizeof(num_rows));
    file.read((char *)&num_cols, sizeof(num_cols));

    num_images = ntohl(num_images);
    num_rows = ntohl(num_rows);
    num_cols = ntohl(num_cols);

    Eigen::MatrixXd images(num_rows * num_cols, num_images);

    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_rows * num_cols; ++j)
        {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            images(j, i) = static_cast<double>(pixel) / 255.0;
        }
    }

    return images;
}

/**
 * @brief Read a MNIST label file
 *
 * @param filename Name of the file to read
 * @return A matrix containing the labels
 */
Eigen::MatrixXd read_mnist_labels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cout << "Error: Failed to open file: " << filename.c_str() << std::endl;
        return Eigen::MatrixXd();
    }

    int magic_number = 0;
    int num_labels = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);

    if (magic_number != 2049)
    {
        std::cout << "Error: Invalid magic number in file: " << filename.c_str();
        return Eigen::MatrixXd();
    }

    file.read((char *)&num_labels, sizeof(num_labels));
    num_labels = ntohl(num_labels);

    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(10, num_labels);

    for (int i = 0; i < num_labels; ++i)
    {
        unsigned char label = 0;
        file.read((char *)&label, sizeof(label));
        int index = static_cast<int>(label);
        labels(index, i) = 1.0;
    }
    return labels;
}

/**
 * @brief Fetch MNIST dataset and save to local directory if it doesn't exist
 * @param data_dir Local directory to save MNIST dataset in
 *
 * @return A tuple containing the training and test sets
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
fetch_mnist(const std::string &data_dir)
{
    const std::string train_images_url = BASE_URL + "train-images-idx3-ubyte.gz";
    const std::string train_images_path = data_dir + "/train-images-idx3-ubyte.gz";
    const std::string train_images_file = data_dir + "/train-images-idx3-ubyte";
    const std::string train_labels_url = BASE_URL + "train-labels-idx1-ubyte.gz";
    const std::string train_labels_path = data_dir + "/train-labels-idx1-ubyte.gz";
    const std::string train_labels_file = data_dir + "/train-labels-idx1-ubyte";
    const std::string test_images_url = BASE_URL + "t10k-images-idx3-ubyte.gz";
    const std::string test_images_path = data_dir + "/t10k-images-idx3-ubyte.gz";
    const std::string test_images_file = data_dir + "/t10k-images-idx3-ubyte";
    const std::string test_labels_url = BASE_URL + "t10k-labels-idx1-ubyte.gz";
    const std::string test_labels_path = data_dir + "/t10k-labels-idx1-ubyte.gz";
    const std::string test_labels_file = data_dir + "/t10k-labels-idx1-ubyte";

    if (!std::filesystem::exists(train_images_file) ||
        !std::filesystem::exists(train_labels_file) ||
        !std::filesystem::exists(test_images_file) ||
        !std::filesystem::exists(test_labels_file))
    {
        std::filesystem::create_directories(data_dir);

        std::cout << "Downloading MNIST training images..." << std::endl;
        download_file(train_images_url, train_images_path);

        std::cout << "Downloading MNIST training labels..." << std::endl;
        download_file(train_labels_url, train_labels_path);

        std::cout << "Downloading MNIST test images..." << std::endl;
        download_file(test_images_url, test_images_path);

        std::cout << "Downloading MNIST test labels..." << std::endl;
        download_file(test_labels_url, test_labels_path);

        std::cout << "Extracting MNIST training images..." << std::endl;
        unzip_file(train_images_path, train_images_file);

        std::cout << "Extracting MNIST training labels..." << std::endl;
        unzip_file(train_labels_path, train_labels_file);

        std::cout << "Extracting MNIST test images..." << std::endl;
        unzip_file(test_images_path, test_images_file);

        std::cout << "Extracting MNIST test labels..." << std::endl;
        unzip_file(test_labels_path, test_labels_file);
    }

    Eigen::MatrixXd x_train = read_mnist_images(train_images_file);
    Eigen::MatrixXd y_train = read_mnist_labels(train_labels_file);
    Eigen::MatrixXd x_test = read_mnist_images(test_images_file);
    Eigen::MatrixXd y_test = read_mnist_labels(test_labels_file);

    return std::make_tuple(x_train, y_train, x_test, y_test);
};