#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <unistd.h>

#include <Eigen/Core>

#include <NNFS/Core>

#include "mnist.cpp"

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> create_data(int samples, int classes)
{
    Eigen::MatrixXd X(samples * classes, 2);
    Eigen::MatrixXd y(samples * classes, classes);
    y.setZero();
    for (int class_number = 0; class_number < classes; ++class_number)
    {
        // auto ix = Eigen::VectorXi::LinSpaced(samples, samples * class_number, samples * (class_number + 1) - 1);
        auto r = Eigen::VectorXd::LinSpaced(samples, 0.0, 1.0);
        auto t = Eigen::VectorXd::LinSpaced(samples, class_number * 4.0, (class_number + 1) * 4.0) + Eigen::VectorXd::Random(samples) * 0.2;
        X.block(samples * class_number, 0, samples, 2) << (r.array() * sin(t.array() * 2.5)).matrix(), (r.array() * cos(t.array() * 2.5)).matrix();
        y.block(samples * class_number, class_number, samples, 1).setConstant(1.0);
    }
    return std::make_pair(X, y);
}

int main()
{
    LOG_INFO("Fetching dataset");

    std::string data_dir = "data";
    auto [x_train, y_train, x_test, y_test] = fetch_mnist(data_dir);

    // int number_of_points = 100; // number of points per class
    // int classes = 3;            // number of classes

    // auto [x_train, y_train] = create_data(number_of_points, classes);
    // auto [x_test, y_test] = create_data(number_of_points, classes);

    // Shuffle data and labels
    // Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(x_train.rows());
    // perm.setIdentity();
    // std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), std::mt19937(std::random_device()()));
    // x_train = perm * x_train;
    // y_train = perm * y_train;

    // Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm2(x_test.rows());
    // perm2.setIdentity();
    // std::shuffle(perm2.indices().data(), perm2.indices().data() + perm2.indices().size(), std::mt19937(std::random_device()()));
    // x_test = perm2 * x_test;
    // y_test = perm2 * y_test;

    // LOG_DEBUG("Shape of training dataset - rows: " << x_train.rows() << " cols: " << x_train.cols());
    // LOG_DEBUG("Shape of training labels - rows: " << y_train.rows() << " cols: " << y_train.cols());
    // LOG_DEBUG("Shape of validation dataset - rows: " << x_test.rows() << " cols: " << x_test.cols());
    // LOG_DEBUG("Shape of validation labels - rows: " << y_test.rows() << " cols: " << y_test.cols());

    // LOG_INFO("Creating model");

    // std::shared_ptr<NNFS::Loss> loss = std::make_shared<NNFS::CCESoftmax>(std::make_shared<NNFS::Softmax>(), std::make_shared<NNFS::CCE>());

    // double learning_rate = 1e-3;
    // double decay = 1e-3;

    // std::shared_ptr<NNFS::Optimizer> optimizer = std::make_shared<NNFS::Adam>(learning_rate, decay); // learning_rate, decay

    // std::shared_ptr<NNFS::NeuralNetwork> model = std::make_shared<NNFS::NeuralNetwork>(loss, optimizer);

    // model->add_layer(std::make_shared<NNFS::Dense>(784, 128));
    // model->add_layer(std::make_shared<NNFS::ReLU>());
    // model->add_layer(std::make_shared<NNFS::Dense>(128, 128));
    // model->add_layer(std::make_shared<NNFS::ReLU>());
    // model->add_layer(std::make_shared<NNFS::Dense>(128, 10));

    // LOG_INFO("Compiling model");

    // model->compile();

    // LOG_INFO("Training model");

    // model->fit(x_train, y_train, x_test, y_test, 100, 128);

    std::string file_path = "MNIST.bin";
    char* home_dir = getenv("HOME");
    file_path = strcat(home_dir,"/MNIST.bin");
    // LOG_INFO("Saving model to file " << file_path);

    // model->save(file_path);

    std::shared_ptr<NNFS::NeuralNetwork> model = std::make_shared<NNFS::NeuralNetwork>();

    LOG_INFO("Loading model from file " << file_path);
    model->load(file_path);

    LOG_INFO("Evaluating model");

    double accuracy;
    model->accuracy(accuracy, x_test, y_test);
    LOG_INFO("Test set accuracy: " << accuracy);

    LOG_INFO("Predicting");

    // Slice first 5 rows of test set
    Eigen::MatrixXd x_test_slice = x_test.topRows(10);

    LOG_INFO(x_test_slice.row(0).reshaped(28,28));
    Eigen::MatrixXd preds = model->predict(x_test_slice);
    LOG_INFO(preds);
    Eigen::VectorXi pred_labels;
    Eigen::VectorXi labels;

    NNFS::Metrics::onehotdecode(labels, y_test.topRows(10));
    NNFS::Metrics::onehotdecode(pred_labels, preds);

    LOG_INFO("First 10 predictions: " << pred_labels.transpose());
    LOG_INFO("First 10 labels:      " << labels.transpose());

    return 0;
}