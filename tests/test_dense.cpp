#include "gtest/gtest.h"

#define LOG_LEVEL LOG_SEV_NONE

#include <NNFS/Core>

class DenseTest : public ::testing::Test
{
protected:
    std::shared_ptr<NNFS::Dense> dense_ = std::make_shared<NNFS::Dense>(4, 3);
};

// Test Dense::backward method
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
    dense_->forward(fout, inputs);
    EXPECT_EQ(fout.rows(), 3);
    EXPECT_EQ(fout.cols(), 3);

    Eigen::MatrixXd out(1, 3);
    dense_->backward(out, dx);

    EXPECT_TRUE(dense_->dbiases().isApprox(expected_dbiases, 1e-7));
    EXPECT_TRUE(dense_->dweights().isApprox(expected_dweights, 1e-7));
}

// Test Dense::weights and Dense::biases methods with with incorrect dimensions
TEST_F(DenseTest, IncorrectDimensionsTest)
{
    Eigen::MatrixXd weights(3, 3);
    Eigen::MatrixXd biases(3, 1);
    EXPECT_THROW(dense_->weights(weights), std::invalid_argument);
    EXPECT_THROW(dense_->biases(biases), std::invalid_argument);
}

// Test Dense::shape and Dense::parameters methods
TEST_F(DenseTest, ShapeAndParametersTest)
{
    int input;
    int output;
    dense_->shape(input, output);
    EXPECT_EQ(input, 4);
    EXPECT_EQ(output, 3);

    EXPECT_EQ(dense_->parameters(), 15);
}

// Test with L1 and L2 regularization
TEST_F(DenseTest, RegularizationTest)
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

    double l1_weights_regularizer = dense_optimization_->l1_weights_regularizer();
    double l2_weights_regularizer = dense_optimization_->l2_weights_regularizer();
    double l1_biases_regularizer = dense_optimization_->l1_biases_regularizer();
    double l2_biases_regularizer = dense_optimization_->l2_biases_regularizer();

    EXPECT_EQ(l1_weights_regularizer, 0.01);
    EXPECT_EQ(l2_weights_regularizer, 0.01);
    EXPECT_EQ(l1_biases_regularizer, 0.01);
    EXPECT_EQ(l2_biases_regularizer, 0.01);

    Eigen::MatrixXd dx{
        {1., 1., 1.},
        {2., 2., 2.},
        {3., 3., 3.},
    };

    Eigen::MatrixXd expected_dbiases{
        {6.012, 6.014, 6.016},
    };

    Eigen::MatrixXd expected_dweights{
        {0.512, 0.514, 0.516},
        {20.118, 20.12, 20.122},
        {10.924, 10.926, 10.928},
        {4.13, 4.132, 4.134},
    };

    Eigen::MatrixXd expected_bout{
        {0.6, 1.5, 2.4, 3.3},
        {1.2, 3, 4.8, 6.6},
        {1.8, 4.5, 7.2, 9.9},
    };

    Eigen::MatrixXd bout;
    dense_optimization_->backward(bout, dx);

    // std::cout << "bout: " << bout << std::endl;
    // std::cout << "dbiases: " << dense_optimization_->dbiases() << std::endl;
    // std::cout << "dweights: " << dense_optimization_->dweights() << std::endl;

    EXPECT_TRUE(dense_optimization_->dbiases().isApprox(expected_dbiases, 1e-7));
    EXPECT_TRUE(dense_optimization_->dweights().isApprox(expected_dweights, 1e-7));
}