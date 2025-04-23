#pragma once

#include <Eigen/Core>

#include <problem/costterm/weightleastsqcostterm.hpp>
#include <problem/optimizationproblem.hpp>
#include <trajectory/bspline/variable.hpp>
#include <trajectory/interface.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace bspline {

        class Interface : public traj::Interface {
            public:
                using Ptr = std::shared_ptr<Interface>;
                using ConstPtr = std::shared_ptr<const Interface>;

                using VeloType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create a shared instance of Interface.
                static Ptr MakeShared(const Time& knot_spacing = Time(0.1));

                //Constructs a B-spline interface.
                Interface(const Time& knot_spacing = Time(0.1));

                //Get velocity interpolator at a specific time.
                Evaluable<VeloType>::ConstPtr getVelocityInterpolator(const Time& time);

                //Adds all state variables to the optimization problem.
                void addStateVariables(Problem& problem) const;

                using KnotMap = std::map<Time, Variable::Ptr>;

                //retrieves the internal knot map
                const KnotMap& knot_map() const { return knot_map_; }

            private:
            
                KnotMap knot_map_;          //Ordered map of knots
            };

        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp