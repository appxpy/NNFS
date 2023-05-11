#include "gtest/gtest.h"
#include <NNFSCore/Loss/CCE.hpp>
#include <NNFSCore/Loss/CCE_Softmax.hpp>
#include <NNFSCore/Activation/Softmax.hpp>
#include <Eigen/Dense>

class CCETest : public ::testing::Test
{
protected:
    NNFSCore::CCE cce_;
};

class CCESoftmaxTest : public ::testing::Test
{
protected:
    std::shared_ptr<NNFSCore::Softmax> softmax_ = std::make_shared<NNFSCore::Softmax>();
    std::shared_ptr<NNFSCore::CCE> cce_ = std::make_shared<NNFSCore::CCE>();
    std::shared_ptr<NNFSCore::CCESoftmax> cce_softmax_ = std::make_shared<NNFSCore::CCESoftmax>(softmax_, cce_);
};

TEST_F(CCETest, ForwardPassTest)
{
    Eigen::MatrixXd y_true(2, 3);
    y_true << 0, 1, 0,
        0, 0, 1;
    Eigen::MatrixXd y_pred(2, 3);
    y_pred << 0.05, 0.95, 0,
        0.1, 0.8, 0.1;
    double loss;
    cce_.calculate(loss, y_pred, y_true);
    ASSERT_NEAR(1.177, loss, 0.01);
}

TEST_F(CCETest, BackwardPassTest)
{
    Eigen::MatrixXd y_true(2, 3); // example one-hot encoded labels with 3 examples and 4 classes
    y_true << 0, 1, 0,
        0, 0, 1;
    Eigen::MatrixXd y_pred(2, 3); // example predicted probabilities with 3 examples and 4 classes
    y_pred << 0.01, 0.95, 0.04,
        0.1, 0.8, 0.1;
    Eigen::MatrixXd out;

    Eigen::MatrixXd expected_out{
        {-0, -0.526316, -0},
        {-0, -0, -5},
    };
    cce_.backward(out, y_pred, y_true);
    ASSERT_TRUE(out.isApprox(expected_out, 1e-7));
}

TEST_F(CCESoftmaxTest, ForwardPassTest)
{
    Eigen::MatrixXd input{
        {1, 2, 3},
    };

    Eigen::MatrixXd labels{
        {0, 0, 1},
    };

    double loss;

    cce_softmax_->calculate(loss, input, labels);

    ASSERT_NEAR(loss, 0.4076059, 1e-7);
}

TEST_F(CCESoftmaxTest, BackwardPassTest)
{
    Eigen::MatrixXd dvalues{
        {.1, .2, .3},
    };

    Eigen::MatrixXd labels{
        {0, 0, 1},
    };

    Eigen::MatrixXd expected{
        {0.1, 0.2, -0.7},
    };

    Eigen::MatrixXd out;

    cce_softmax_->backward(out, dvalues, labels);

    ASSERT_TRUE(out.isApprox(expected, 1e-7));
}

TEST_F(CCESoftmaxTest, IntegrityBackwardPassTest)
{
    Eigen::MatrixXd softmax_outputs{
        {0.7, 0.1, 0.2},
        {0.1, 0.5, 0.4},
        {0.02, 0.9, 0.08},
    };
    Eigen::MatrixXd labels{
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
    };

    Eigen::MatrixXd ccesout;

    cce_softmax_->backward(ccesout, softmax_outputs, labels);

    Eigen::MatrixXd cceout;

    softmax_->_forward_output = softmax_outputs;

    cce_->backward(cceout, softmax_outputs, labels);

    Eigen::MatrixXd sout;
    softmax_->backward(sout, cceout);

    ASSERT_TRUE(ccesout.isApprox(sout, 1e-7));
}