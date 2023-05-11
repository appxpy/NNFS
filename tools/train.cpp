#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>

#include <Eigen/Core>

#include <NNFSCore/Core>

#include "mnist.cpp"

// void spiral_data(Eigen::MatrixXd &x, Eigen::MatrixXd &y, int number_of_points, int classes)
// {
//     Eigen::VectorXi y_vec(number_of_points * classes);

//     for (int j = 0; j < classes; j++)
//     {
//         Eigen::VectorXi ix(number_of_points);
//         for (int i = 0; i < number_of_points; i++)
//         {
//             ix(i) = i + j * number_of_points;
//         }
//         double r = 0.0;
//         double t = 0.0;
//         for (int i = 0; i < number_of_points; i++)
//         {
//             r = (double)i / number_of_points;
//             t = j * 4 + (double)i / number_of_points * 4 + ((double)rand() / RAND_MAX) * 0.2;
//             x.row(ix(i)) << r * sin(t * 2.5), r * cos(t * 2.5);
//             y_vec(ix(i)) = j;
//         }
//     }

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
    std::cout << "Fetching dataset" << std::endl;

    std::string data_dir = "data";
    auto [x_train, y_train, x_test, y_test] = fetch_mnist(data_dir);

    // Eigen::MatrixXd x_train{{1.0, 2.0, 3.0, 2.5},
    //                         {2.0, 5.0, -1.0, 2.0},
    //                         {-1.5, 2.7, 3.3, -0.8}};

    // Eigen::MatrixXd y_train{
    //     {1.0}, {1.0}, {1.0}};

    // Eigen::MatrixXd x_test{{1.0, 2.0, 3.0, 2.5},
    //                        {2.0, 5.0, -1.0, 2.0},
    //                        {-1.5, 2.7, 3.3, -0.8}};

    // Eigen::MatrixXd y_test{
    //     {1.0}, {1.0}, {1.0}};

    // int number_of_points = 1000; // number of points per class
    // int classes = 3;             // number of classes

    // auto [x_train, y_train] = create_data(number_of_points, classes);
    // auto [x_test, y_test] = create_data(number_of_points, classes);

    std::cout << "x_train_shape rows: " << x_train.rows() << " cols: " << x_train.cols() << std::endl;
    // std::cout << "x_train" << std::endl
    //           << x_train << std::endl;

    // for (int row = 0; row < x_train.rows(); row++)
    // {
    //     for (int col = 0; col < x_train.cols(); col++)
    //     {
    //         std::cout << x_train(row, col) << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "y_train_shape rows: " << y_train.rows() << " cols: " << y_train.cols() << std::endl;
    // std::cout << "y_train" << std::endl
    //           << y_train << std::endl;
    std::cout << "x_test_shape rows: " << x_test.rows() << " cols: " << x_test.cols() << std::endl;
    std::cout << "y_test_shape rows: " << y_test.rows() << " cols: " << y_test.cols() << std::endl;

    //     // Eigen::MatrixXd x_train(2, 3);
    //     // Eigen::MatrixXd y_train(2, 3);

    //     // x_train << 0.6, 0.3, 0.2,
    //     //     0.1, 0.8, 0.5;
    //     // y_train << 1, 0, 0,
    //     //     0, 1, 1;
    std::cout << "Creating model" << std::endl;

    std::vector<std::shared_ptr<NNFSCore::Layer>> layers = {
        std::make_shared<NNFSCore::Dense>(784, 512, 0, 0, 5e-4, 5e-4),
        // std::make_shared<NNFSCore::ReLU>(),
        // std::make_shared<NNFSCore::Dense>(128, 64),
        std::make_shared<NNFSCore::ReLU>(),
        std::make_shared<NNFSCore::Dense>(512, 10),
    };

    std::shared_ptr<NNFSCore::Loss> loss = std::make_shared<NNFSCore::CCESoftmax>(std::make_shared<NNFSCore::Softmax>(), std::make_shared<NNFSCore::CCE>());
    double learning_rate = .02;
    double decay = 5e-7;
    // double momentum = .9;
    // double epsilon = 1e-4;

    std::shared_ptr<NNFSCore::Optimizer> optimizer = std::make_shared<NNFSCore::Adam>(learning_rate, decay); // learning_rate, decay
    // std::shared_ptr<NNFSCore::Optimizer>
    //     optimizer = std::make_shared<NNFSCore::Adagrad>(learning_rate, decay, epsilon);

    std::shared_ptr<NNFSCore::NeuralNetwork>
        model = std::make_shared<NNFSCore::NeuralNetwork>(
            layers, loss, optimizer);
    //     for (const auto &layer_tuple : model->get_layers())
    //     {
    //         // Extract the shared pointers from the tuple
    //         const auto &layer_ptr = std::get<0>(layer_tuple);
    //         const auto &activation_ptr = std::get<1>(layer_tuple);

    //         // Access the members of the objects pointed to by the shared pointers
    //         std::cout << "Layer with " << layer_ptr->output().cols() << " units, activated with " << typeid(*activation_ptr).name() << std::endl;
    //     }

    std::cout << "Training model" << std::endl;

    model->fit(x_train, y_train, x_test, y_test, 20000, 128);

    std::cout << "Evaluating trained model" << std::endl;

    // double loss_value = model->evaluate(x_test, y_test);

    // std::cout << "Validation loss: " << loss_value << std::endl;

    // Eigen::MatrixXd preds = model->predict(x_test);

    // std::cout << "First 5 predictions: \n"
    //           << preds << std::endl;

    // std::cout << "First 5 labels: \n"
    //           << y_test << std::endl;

    // double accuracy = (preds.array() == y_test.array()).mean();

    // std::cout << "Test set accuracy: " << accuracy << std::endl;

    // return 0;
}
