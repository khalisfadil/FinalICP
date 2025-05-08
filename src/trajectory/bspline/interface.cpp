#include <trajectory/bspline/interface.hpp>

#include <evaluable/se3/evaluables.hpp>
#include <evaluable/vspace/evaluables.hpp>
#include <problem/lossfunc/lossfunc.hpp>
#include <problem/noisemodel/staticnoisemodel.hpp>
#include <trajectory/bspline/velocityinterpolator.hpp>

namespace finalicp {
    namespace traj {
        namespace bspline {
            
            auto Interface::MakeShared(const Time& knot_spacing) -> Ptr {
                return std::make_shared<Interface>(knot_spacing);
            }

            Interface::Interface(const Time& knot_spacing) : knot_spacing_(knot_spacing) {}

            void Interface::addStateVariables(Problem& problem) const {
                for (const auto& pair : knot_map_)
                problem.addStateVariable(pair.second->getC());
            }

            auto Interface::getVelocityInterpolator(const Time& time)-> Evaluable<VeloType>::ConstPtr {
                int64_t t2_nano = knot_spacing_.nanosecs() * std::floor(time.nanosecs() / knot_spacing_.nanosecs());
                Time t2(t2_nano);
                Time t1 = t2 - knot_spacing_;
                Time t3 = t2 + knot_spacing_;
                Time t4 = t3 + knot_spacing_;

                // clang-format off
                const auto v1 = knot_map_.try_emplace(t1, Variable::MakeShared(t1, vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero()))).first->second;
                const auto v2 = knot_map_.try_emplace(t2, Variable::MakeShared(t2, vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero()))).first->second;
                const auto v3 = knot_map_.try_emplace(t3, Variable::MakeShared(t3, vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero()))).first->second;
                const auto v4 = knot_map_.try_emplace(t4, Variable::MakeShared(t4, vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero()))).first->second;
                // clang-format on

                // Debug output: Print the StateKey for each control point
                std::cout << "Velocity interpolator at time " << time.nanosecs() / 1e9 << "s uses keys: "
                        << v1->getC()->key() << " "
                        << v2->getC()->key() << " "
                        << v3->getC()->key() << " "
                        << v4->getC()->key() << std::endl;

                std::cout << "Knot times for time " << time.nanosecs() / 1e9 << "s: "
                        << t1.nanosecs() / 1e9 << "s, "
                        << t2.nanosecs() / 1e9 << "s, "
                        << t3.nanosecs() / 1e9 << "s, "
                        << t4.nanosecs() / 1e9 << "s" << std::endl;

                return VelocityInterpolator::MakeShared(time, v1, v2, v3, v4);
            }
        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp