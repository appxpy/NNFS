#pragma once

#include <Eigen/Dense>
#include "../Utilities/clue.hpp"
#include "../Layer/Dense.hpp"

namespace NNFS
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
         * @param decay Learning rate decay (default: 0.0)
         */
        Optimizer(double lr, double decay) : _lr(lr), _current_lr(lr), _iterations(0), _decay(decay) {}

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
        void pre_update_params()
        {
            if (_decay > 0)
            {
                _current_lr = _lr * (1. / (1. + _decay * _iterations));
            }
        }

        /**
         * @brief Post-update parameters (e.g. increase iteration count)
         */
        void post_update_params()
        {
            _iterations += 1;
        }

        /**
         * @brief Get the current learning rate
         *
         * @return double Current learning rate
         */
        double &current_lr()
        {
            return _current_lr;
        }

        /**
         * @brief Get current iteration count
         *
         * @return int Current iteration count
         */
        int &iterations()
        {
            return _iterations;
        }

    protected:
        const double _lr;   // Learning rate (constant)
        double _current_lr; // Current learning rate
        int _iterations;    // Iteration count
        double _decay;      // Learning rate decay
    };
} // namespace NNFS
