#include "gtest/gtest.h"

#define LOG_LEVEL LOG_SEV_NONE

#include <NNFS/Core>

class CCETest : public ::testing::Test
{
protected:
    NNFS::CCE cce_;
};

class CCESoftmaxTest : public ::testing::Test
{
protected:
    std::shared_ptr<NNFS::Softmax> softmax_ = std::make_shared<NNFS::Softmax>();
    std::shared_ptr<NNFS::CCE> cce_ = std::make_shared<NNFS::CCE>();
    std::shared_ptr<NNFS::CCESoftmax> cce_softmax_ = std::make_shared<NNFS::CCESoftmax>(softmax_, cce_);
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
    EXPECT_NEAR(1.177, loss, 0.01);
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
    EXPECT_TRUE(out.isApprox(expected_out, 1e-7));
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

    EXPECT_NEAR(loss, 0.4076059, 1e-7);
}

TEST_F(CCESoftmaxTest, BackwardPassTest)
{
    Eigen::MatrixXd input{
        {1, 2, 3},
    };

    Eigen::MatrixXd dvalues{
        {.1, .2, .3},
    };

    Eigen::MatrixXd labels{
        {0, 0, 1},
    };

    double loss;

    cce_softmax_->calculate(loss, input, labels);

    EXPECT_NEAR(loss, 0.4076059, 1e-7);

    Eigen::MatrixXd expected{
        {0.0900306, 0.244728, -0.334759},
    };

    Eigen::MatrixXd out;

    cce_softmax_->backward(out, dvalues, labels);

    EXPECT_TRUE(out.isApprox(expected, 1e-5));
}

TEST_F(CCESoftmaxTest, IntegrityTest)
{
    Eigen::MatrixXd input{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    };

    Eigen::MatrixXd labels{
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
    };

    double loss;

    cce_softmax_->calculate(loss, input, labels);

    EXPECT_NEAR(loss, 1.40761, 1e-5);

    Eigen::MatrixXd softmax_expected{
        {0.0900306, 0.244728, 0.665241},
        {0.0900306, 0.244728, 0.665241},
        {0.0900306, 0.244728, 0.665241},
    };
    EXPECT_TRUE(cce_softmax_->softmax_out().isApprox(softmax_expected, 1e-5));

    Eigen::MatrixXd ccesout;

    cce_softmax_->backward(ccesout, softmax_expected, labels);

    Eigen::MatrixXd cceout;

    softmax_->_forward_output = softmax_expected;

    cce_->backward(cceout, softmax_expected, labels);

    Eigen::MatrixXd sout;
    softmax_->backward(sout, cceout);

    EXPECT_TRUE(ccesout.isApprox(sout, 1e-4));
}

TEST_F(CCESoftmaxTest, RegularizationLossTest)
{
    std::shared_ptr<NNFS::Dense> dense_optimization_ = std::make_shared<NNFS::Dense>(4, 3, 0.01, 0.01, 0.01, 0.01);
    Eigen::MatrixXd weights(4, 3);
    weights << 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2;
    Eigen::MatrixXd biases(1, 3);
    biases << 0.1, 0.2, 0.3;
    dense_optimization_->weights(weights);
    dense_optimization_->biases(biases);

    Eigen::MatrixXd inputs(3, 4);
    inputs << 1, 2, 3, 2.5,
        2., 5., -1., 2,
        -1.5, 2.7, 3.3, -0.8;

    Eigen::MatrixXd expected_out{
        {5.6, 6.55, 7.5},
        {3.6, 4.5, 5.4},
        {2.54, 3.01, 3.48},
    };

    Eigen::MatrixXd out;
    dense_optimization_->forward(out, inputs);

    EXPECT_TRUE(out.isApprox(expected_out, 1e-7));

    double loss = cce_softmax_->regularization_loss(dense_optimization_);

    EXPECT_NEAR(loss, 0.222, 1e-3);
}