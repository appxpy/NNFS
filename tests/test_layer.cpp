#include <gtest/gtest.h>
#include <Eigen/Core>
#include <iostream>

#include <NNFSCore/Layer/Dense.hpp>

using namespace NNFSCore;

class DenseTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        num_units_ = 10;
        layer_ = std::make_unique<Dense>(num_units_);
    }
    int num_units_;
    std::unique_ptr<Dense> layer_;
};

// Test case for build function
TEST_F(DenseTest, DenseBuild)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 8);
    layer_->build(input);

    // Check that the weights matrix is of the correct shape
    EXPECT_EQ(layer_->weights().rows(), num_units_);
    EXPECT_EQ(layer_->weights().cols(), input.rows());

    // Check that the bias matrix is of the correct shape
    EXPECT_EQ(layer_->bias().rows(), num_units_);
    EXPECT_EQ(layer_->bias().cols(), 1);

    // Check that the gradients matrices are of the correct shape
    EXPECT_EQ(layer_->grad_weights().rows(), num_units_);
    EXPECT_EQ(layer_->grad_weights().cols(), input.rows());
    EXPECT_EQ(layer_->grad_bias().rows(), num_units_);
    EXPECT_EQ(layer_->grad_bias().cols(), 1);
}

// Test case for grad_weights and grad_bias functions
TEST_F(DenseTest, DenseGradients)
{
    // Set some arbitrary values for the gradients
    Eigen::MatrixXd dw = Eigen::MatrixXd::Random(num_units_, 8);
    Eigen::MatrixXd db = Eigen::MatrixXd::Random(num_units_, 1);

    layer_->grad_weights(dw);
    layer_->grad_bias(db);

    // Check that the gradients are set correctly
    EXPECT_EQ(layer_->grad_weights().isApprox(dw), true);
    EXPECT_EQ(layer_->grad_bias().isApprox(db), true);
}

// Test case for update function
TEST_F(DenseTest, DenseUpdate)
{
    // Set up input tensor and expected output tensor
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 8);
    Eigen::MatrixXd output = (*layer_)(input);

    // Set up gradients and learning rate
    Eigen::MatrixXd grad_weights = Eigen::MatrixXd::Random(layer_->weights().rows(), layer_->weights().cols());
    Eigen::MatrixXd grad_bias = Eigen::MatrixXd::Random(layer_->bias().rows(), layer_->bias().cols());
    double lr = 0.01;

    // Update the layer's parameters
    layer_->grad_weights(grad_weights);
    layer_->grad_bias(grad_bias);
    layer_->update(lr);

    // Compute the new output tensor
    Eigen::MatrixXd new_output = (*layer_)(input);

    ASSERT_EQ(new_output, layer_->output());

    // Compute the expected output tensor using the updated weights and biases
    Eigen::MatrixXd expected_output = output - lr * (grad_weights * input + grad_bias.replicate(1, input.cols()));

    // Check that the new output tensor matches the expected output tensor
    ASSERT_EQ(new_output.rows(), expected_output.rows());
    ASSERT_EQ(new_output.cols(), expected_output.cols());
    for (int i = 0; i < new_output.rows(); i++)
    {
        for (int j = 0; j < new_output.cols(); j++)
        {
            EXPECT_NEAR(new_output(i, j), expected_output(i, j), 1e-6);
        }
    }
}