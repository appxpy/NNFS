#include "gtest/gtest.h"
#include <NNFSCore/Layer/Dense.hpp>
#include <Eigen/Dense>
#define LOG_LEVEL_
class DenseTest : public ::testing::Test
{
protected:
    std::shared_ptr<NNFSCore::Dense> linear_ = std::make_shared<NNFSCore::Dense>(4, 3);
};

TEST_F(DenseTest, BackwardPassTest)
{
    Eigen::MatrixXd dx{
        {1., 1., 1.},
        {2., 2., 2.},
        {3., 3., 3.},
    };

    Eigen::MatrixXd inputs{
        {1, 2, 3, 2.5},
        {2., 5., -1., 2},
        {-1.5, 2.7, 3.3, -0.8},
    };

    Eigen::MatrixXd expected_dbiases{
        {6.0, 6.0, 6.0},
    };

    Eigen::MatrixXd expected_dweights{
        {0.5, 0.5, 0.5},
        {20.1, 20.1, 20.1},
        {10.9, 10.9, 10.9},
        {4.1, 4.1, 4.1},
    };

    Eigen::MatrixXd fout;
    linear_->forward(fout, inputs);
    ASSERT_EQ(fout.rows(), 3);
    ASSERT_EQ(fout.cols(), 3);

    Eigen::MatrixXd out(1, 3);
    linear_->backward(out, dx);

    ASSERT_TRUE(linear_->dbiases().isApprox(expected_dbiases, 1e-7));
    ASSERT_TRUE(linear_->dweights().isApprox(expected_dweights, 1e-7));
}