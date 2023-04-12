#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Core>

#include <NNFSCore/Core>
#include <NNFSCore/Model/NeuralNetwork.hpp>

#include "mnist.cpp"

int main()
{
    std::cout << "Fetching dataset" << std::endl;

    std::string data_dir = "data";
    auto [x_train, y_train, x_test, y_test] = fetch_mnist(data_dir);

    std::cout << "x_train_shape rows: " << x_train.rows() << " cols: " << x_train.cols() << std::endl;
    std::cout << "y_train_shape rows: " << y_train.rows() << " cols: " << y_train.cols() << std::endl;
    std::cout << "x_test_shape rows: " << x_test.rows() << " cols: " << x_test.cols() << std::endl;
    std::cout << "y_test_shape rows: " << y_test.rows() << " cols: " << y_test.cols() << std::endl;

    std::cout << "Creating model" << std::endl;

    std::vector<std::tuple<std::shared_ptr<NNFSCore::Layer>, std::shared_ptr<NNFSCore::Activation>>> layers = {
        {std::make_shared<NNFSCore::Dense>(784), std::make_shared<NNFSCore::ReLU>()},
        {std::make_shared<NNFSCore::Dense>(500), std::make_shared<NNFSCore::ReLU>()},
        {std::make_shared<NNFSCore::Dense>(10), std::make_shared<NNFSCore::Sigmoid>()}};

    std::shared_ptr<NNFSCore::Loss> loss = std::make_shared<NNFSCore::BinaryCrossEntropy>();
    double learning_rate = 1.0;
    double regularization_factor = 2.0;

    auto model = std::make_shared<NNFSCore::NeuralNetwork>(
        layers, loss, learning_rate, regularization_factor);

    std::cout << "Training model" << std::endl;

    model->fit(x_train, y_train, 1, 128, true);

    std::cout << "Evaluating trained model" << std::endl;

    double loss_value = model->evaluate(x_test, y_test);

    std::cout << "Validation loss: " << loss_value << std::endl;

    Eigen::MatrixXd preds = model->predict(x_test);

    std::cout << "First 5 predictions: \n"
              << preds.block(0, 0, 1, 1000) << std::endl;

    std::cout << "First 5 labels: \n"
              << y_test.block(0, 0, 1, 1000) << std::endl;

    double accuracy = (preds.array() == y_test.array()).mean();

    std::cout << "Test set accuracy: " << accuracy << std::endl;

    return 0;
}
