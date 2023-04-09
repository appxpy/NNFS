#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <NNFSCore/Activation/ReLU.hpp>
#include <NNFSCore/Activation/Sigmoid.hpp>
#include <NNFSCore/Activation/Linear.hpp>

TEST(ActivationTest, TestReLU)
{
    Eigen::MatrixXd input(1, 2);
    input << -1, 1;
    Eigen::MatrixXd output(1, 2);
    output << 0, 1;

    NNFSCore::ReLU relu;

    // Checking shapes
    ASSERT_EQ(relu(Eigen::MatrixXd::Random(10, 20)).rows(), 10);
    ASSERT_EQ(relu(Eigen::MatrixXd::Random(10, 20)).cols(), 20);
    // Checking values
    ASSERT_TRUE(relu(input).isApprox(output));
    ASSERT_TRUE(relu.gradient(input).isApprox(output));
}

TEST(ActivationTest, TestSigmoid)
{
    Eigen::MatrixXd input(1, 3);
    input << -100, 0, 100;

    NNFSCore::Sigmoid sigmoid;

    // Checking shapes
    ASSERT_EQ(sigmoid(Eigen::MatrixXd::Random(10, 20)).rows(), 10);
    ASSERT_EQ(sigmoid(Eigen::MatrixXd::Random(10, 20)).cols(), 20);
    // Checking values
    ASSERT_NEAR(sigmoid(input)(0, 0), 0.0, 1e-6);
    ASSERT_NEAR(sigmoid(input)(0, 1), 0.5, 1e-6);
    ASSERT_NEAR(sigmoid(input)(0, 2), 1.0, 1e-6);
    ASSERT_NEAR(sigmoid.gradient(input)(0, 0), 0.0, 1e-6);
    ASSERT_NEAR(sigmoid.gradient(input)(0, 2), 0.0, 1e-6);
}

TEST(ActivationTest, TestLinear)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(3, 3);

    NNFSCore::Linear linear;

    // Checking shapes
    ASSERT_EQ(linear(Eigen::MatrixXd::Random(10, 20)).rows(), 10);
    ASSERT_EQ(linear(Eigen::MatrixXd::Random(10, 20)).cols(), 20);

    // Checking values
    Eigen::MatrixXd output = linear(input);
    ASSERT_TRUE(output.isApprox(input));

    // Checking gradient values
    Eigen::MatrixXd gradient_output = linear.gradient(input);
    ASSERT_TRUE(gradient_output.isApprox(Eigen::MatrixXd::Ones(input.rows(), input.cols())));
}