#pragma once

#include <chrono>
#include <iostream>

namespace math {
    namespace common {

        // -----------------------------------------------------------------------------
        /**
         * \brief **High-precision wall timer class for benchmarking functions.**
         * \details Uses `std::chrono::high_resolution_clock` for accurate time measurements.
         */
        class Timer {

            public:

                // -----------------------------------------------------------------------------
                /** \brief Default constructor (initializes and starts timer). */
                Timer() { reset(); }

                // -----------------------------------------------------------------------------
                /** \brief Resets the timer to the current time. */
                void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }

                // -----------------------------------------------------------------------------
                /** \brief Returns **elapsed time in seconds** since last reset. */
                double seconds() const {
                    return get_elapsed_time<std::chrono::seconds>();
                }

                // -----------------------------------------------------------------------------
                /** \brief Returns **elapsed time in milliseconds** since last reset. */
                double milliseconds() const {
                    return get_elapsed_time<std::chrono::milliseconds>();
                }

                // -----------------------------------------------------------------------------
                /** \brief Returns **elapsed time in microseconds** since last reset. */
                double microseconds() const {
                    return get_elapsed_time<std::chrono::microseconds>();
                }

                // -----------------------------------------------------------------------------
                /** \brief Returns **elapsed time in nanoseconds** since last reset. */
                double nanoseconds() const {
                    return get_elapsed_time<std::chrono::nanoseconds>();
                }

            private:

                // -----------------------------------------------------------------------------
                /** \brief Stores the start time when `reset()` is called. */
                std::chrono::high_resolution_clock::time_point start_time_;

                // -----------------------------------------------------------------------------
                /** \brief Helper function to compute elapsed time in different units. */
                template <typename Duration>
                double get_elapsed_time() const {
                    auto elapsed = std::chrono::high_resolution_clock::now() - start_time_;
                    return std::chrono::duration_cast<Duration>(elapsed).count();
                }
        };

    }  // namespace common
} // math
