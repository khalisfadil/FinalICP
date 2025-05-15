#pragma once

#include <matrixoperator/matrix.hpp>
#include <matrixoperator/matrixsparse.hpp>
#include <matrixoperator/vector.hpp>

#include <common/timer.hpp>

#include <evaluable/evaluable.hpp>
#include <evaluable/statevar.hpp>

#include <evaluable/imu/evaluables.hpp>
#include <evaluable/p2p/evaluables.hpp>
#include <evaluable/se3/evaluables.hpp>
#include <evaluable/vspace/evaluables.hpp>

#include <problem/costterm/weightleastsqcostterm.hpp>
#include <problem/lossfunc/lossfunc.hpp>
#include <problem/noisemodel/staticnoisemodel.hpp>
#include <problem/noisemodel/dynamicnoisemodel.hpp>
#include <problem/optimizationproblem.hpp>
#include <problem/slidingwindowfilter.hpp>

#include <solver/covariance.hpp>
#include <solver/dogleggaussnewtonsolver.hpp>
#include <solver/gausnewtonsolver.hpp>
#include <solver/gausnewtonsolvernva.hpp>
#include <solver/levmarqgaussnewtonsolver.hpp>
#include <solver/linesearchgaussnewtonsolver.hpp>

#include <trajectory/bspline/interface.hpp>
#include <trajectory/constacc/interface.hpp>
#include <trajectory/constvel/interface.hpp>
#include <trajectory/singer/interface.hpp>
