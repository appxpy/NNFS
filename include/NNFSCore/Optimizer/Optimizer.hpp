#ifndef OPTIMIZER_BASE_HPP
#define OPTIMIZER_BASE_HPP

#include <Eigen/Dense>
#include "../Utilities/clue.hpp"
#include "../Layer/Dense.hpp"

namespace NNFSCore
{
    class Optimizer
    {
    public:
        Optimizer(double lr) : _lr(lr), _current_lr(lr) {}

        virtual ~Optimizer() = default;

        virtual void update_params(std::shared_ptr<Dense> &layer) = 0;

        // virtual void update_params(Eigen::MatrixXd &weights, Eigen::MatrixXd &biases, Eigen::MatrixXd &dweights, Eigen::MatrixXd &dbiases) = 0;

        virtual void pre_update_params() = 0;

        virtual void post_update_params() = 0;

        double &current_lr()
        {
            return _current_lr;
        }

    protected:
        const double _lr;
        double _current_lr;
    };
}

#endif