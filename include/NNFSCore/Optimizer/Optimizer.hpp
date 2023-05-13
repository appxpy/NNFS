#pragma once

#include <Eigen/Dense>
#include "../Utilities/clue.hpp"
#include "../Layer/Dense.hpp"

namespace NNFSCore
{
    /**
     * @brief Base class for all optimizers
     *
     * @details This class is the base class for all optimizers. It provides the interface for all optimizers.
     */
    class Optimizer
    {
    public:
        /**
         * @brief Construct a new Optimizer object
         *
         * @param lr Learning rate
         */
        Optimizer(double lr) : _lr(lr), _current_lr(lr) {}

        /**
         * @brief Basic destructor
         */
        virtual ~Optimizer() = default;

        /**
         * @brief Update the parameters of the layer
         *
         * @param[in,out] layer Layer to update
         */
        virtual void update_params(std::shared_ptr<Dense> &layer) = 0;

        /**
         * @brief Pre-update parameters (e.g. learning rate decay)
         */
        virtual void pre_update_params() = 0;

        /**
         * @brief Post-update parameters (e.g. increase iteration count)
         */
        virtual void post_update_params() = 0;

        /**
         * @brief Get the current learning rate
         *
         * @return double Current learning rate
         */
        double &current_lr()
        {
            return _current_lr;
        }

    protected:
        const double _lr;   // Learning rate (constant)
        double _current_lr; // Current learning rate
    };
} // namespace NNFSCore
