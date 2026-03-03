#pragma once

#include "stdfin/random/threefry_engine.hpp"

namespace im {

using RNG = stdfin::threefry_13_64;
using seed_type = RNG::result_type;

} // namespace im

using RNG = im::RNG;
using seed_type = im::seed_type;
