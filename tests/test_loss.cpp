#include "gtest/gtest.h"
#include <NNFSCore/Loss/BinaryCrossEntropy.hpp>
#include <Eigen/Dense>

class LossTest : public ::testing::Test
{
protected:
    NNFSCore::BinaryCrossEntropy bce_;
};

TEST_F(LossTest, GradientDescent)
{
    int num_examples = 100;
    int num_features = 20;
    Eigen::MatrixXd predictions = Eigen::MatrixXd::Random(num_examples, num_features).cwiseAbs();
    predictions = predictions / predictions.maxCoeff();
    Eigen::MatrixXd labels = Eigen::MatrixXd::Constant(num_examples, num_features, 0.99);

    Eigen::MatrixXd output_high = bce_.operator()(predictions, labels);
    Eigen::MatrixXd output_low = bce_.operator()(labels, labels);

    EXPECT_GT(output_high.mean(), output_low.mean());
    EXPECT_GT(0.1, output_low.mean());
}

TEST_F(LossTest, GradientTest)
{
    int num_examples = 100;
    int num_features = 20;
    Eigen::MatrixXd predictions = Eigen::MatrixXd::Random(num_examples, num_features).cwiseAbs();
    predictions = predictions / predictions.maxCoeff();
    Eigen::MatrixXd labels = Eigen::MatrixXd::Constant(num_examples, num_features, 0.99);

    Eigen::MatrixXd gradient_result = bce_.gradient(predictions, labels);

    EXPECT_EQ(gradient_result.rows(), num_examples);
    EXPECT_EQ(gradient_result.cols(), num_features);

    for (int i = 0; i < gradient_result.rows(); ++i)
    {
        for (int j = 0; j < gradient_result.cols(); ++j)
        {
            EXPECT_GE(gradient_result(i, j), -1e6);
            EXPECT_LE(gradient_result(i, j), 1e6);
        }
    }
}
