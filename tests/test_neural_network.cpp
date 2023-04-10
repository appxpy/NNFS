#include "gtest/gtest.h"
#include <NNFSCore/Model/NeuralNetwork.hpp>
#include <Eigen/Dense>

class NeuralNetworkTest : public ::testing::Test
{
protected:
    NNFSCore::NeuralNetwork nn_;
};
