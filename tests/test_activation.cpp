#include "gtest/gtest.h"
#include <NNFSCore/Core>

class ReLUTest : public ::testing::Test
{
protected:
    NNFSCore::ReLU ReLU_;
};

TEST_F(ReLUTest, GeneralTest)
{
    Eigen::MatrixXd x{
        {1, 2, -3, -4},
        {2, -7, -1, 3},
        {-1, 2, 5, -1},
    };
    Eigen::MatrixXd x_out;

    Eigen::MatrixXd x_expected{
        {1, 2, 0, 0},
        {2, 0, 0, 3},
        {0, 2, 5, 0},
    };
    ReLU_.forward(x_out, x);

    ASSERT_TRUE(x_out.isApprox(x_expected, 1e-7));

    Eigen::MatrixXd dx_out;

    Eigen::MatrixXd dx_expected{
        {1, 2, 0, 0},
        {2, 0, 0, 3},
        {0, 2, 5, 0},
    };

    ReLU_.backward(dx_out, x);

    ASSERT_TRUE(dx_out.isApprox(dx_expected, 1e-7));
}

class SoftmaxTest : public ::testing::Test
{
protected:
    NNFSCore::Softmax Softmax_;
};

TEST_F(SoftmaxTest, GeneralTest)
{
    Eigen::MatrixXd x{
        {.1, .2, .3},
    };
    Eigen::MatrixXd x_out;

    Eigen::MatrixXd x_expected{
        {0.30060961, 0.33222499, 0.3671654},
    };
    Softmax_.forward(x_out, x);

    ASSERT_TRUE(x_out.isApprox(x_expected, 1e-7));

    Eigen::MatrixXd x_custom_out{
        {0.7, 0.2, 0.1},
    };

    Softmax_._forward_output = x_custom_out;

    Eigen::MatrixXd dx_out;

    Eigen::MatrixXd dx_expected{
        {-0.028, 0.012, 0.016}};

    Softmax_.backward(dx_out, x);

    ASSERT_TRUE(dx_out.isApprox(dx_expected, 1e-7));
}