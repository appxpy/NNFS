#include <gtest/gtest.h>

#include <iostream>
#include <NNFSCore/Activation/Linear.hpp>
#include <NNFSCore/Activation/ReLU.hpp>

TEST(ActivationTest, ActivationLinearGradientTest)
{
    NNFSCore::Linear testLinear;
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
        3, 4;

    Eigen::MatrixXd expected_output(2, 2);
    expected_output << 1, 1,
        1, 1;

    Eigen::MatrixXd output = testLinear.gradient(input);

    ASSERT_TRUE(output.isApprox(expected_output));
}

TEST(Activation1Test, ActivationLinearOperatorTest)
{
    NNFSCore::Linear testLinear;
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
        3, 4;

    Eigen::MatrixXd expected_output(2, 2);
    expected_output << 1, 2,
        3, 4;

    Eigen::MatrixXd output = testLinear(input);

    ASSERT_TRUE(output.isApprox(expected_output));
}

TEST(Activation2Test, ActivationReLUOperatorTest)
{
    NNFSCore::ReLU testReLU;
    Eigen::MatrixXd input(2, 2);
    input << 1, -1,
        3, -4;

    Eigen::MatrixXd expected_output(2, 2);
    expected_output << 1, 0,
        3, 0;

    Eigen::MatrixXd output = testReLU(input);

    ASSERT_TRUE(output.isApprox(expected_output));
}

TEST(Activation3Test, ActivationReLUGradientTest)
{
    NNFSCore::ReLU testReLU;
    Eigen::MatrixXd input(2, 2);
    input << 1, -1,
        3, -4;

    Eigen::MatrixXd expected_output(2, 2);
    expected_output << 1, 0,
        1, 0;

    Eigen::MatrixXd output = testReLU.gradient(input);

    ASSERT_TRUE(output.isApprox(expected_output));
}