#pragma once

#include <cstdint>
#include <functional>

namespace finalicp{
    namespace traj {
        //Represents a high-precision timestamp using nanoseconds.
        class Time {
            public:
                //Default constructor, initializes time to zero.
                Time() : nsecs_(0) {}

                //Constructs a time object from nanoseconds.
                explicit Time(int64_t nsecs) : nsecs_(nsecs) {}

                //Constructs a time object from seconds.
                explicit Time(double secs) : nsecs_(secs * 1e9) {}

                //Constructs a time object from separate seconds and nanoseconds.
                Time(int32_t secs, int32_t nsec) {
                    nsecs_ = static_cast<int64_t>(secs) * 1'000'000'000 + static_cast<int64_t>(nsec);
                }

                //get time in seconds (double precision).
                double seconds() const { return static_cast<double>(nsecs_) * 1e-9; }

                //Retrieves the time in nanoseconds.
                const int64_t& nanosecs() const { return nsecs_; }

                //Adds another `Time` object to this one.
                Time& operator+=(const Time& other) {
                    nsecs_ += other.nsecs_;
                    return *this;
                }

                //Adds two `Time` objects.
                Time operator+(const Time& other) const {
                    Time temp(*this);
                    temp += other;
                    return temp;
                }

                //Subtracts another `Time` object from this one.
                Time& operator-=(const Time& other) {
                    nsecs_ -= other.nsecs_;
                    return *this;
                }

                //Subtracts two `Time` objects.
                Time operator-(const Time& other) const {
                    Time temp(*this);
                    temp -= other;
                    return temp;
                }

                //Equality comparison operator.
                bool operator==(const Time& other) const { return nsecs_ == other.nsecs_; }

                //Inequality comparison operator.
                bool operator!=(const Time& other) const { return nsecs_ != other.nsecs_; }

                //Less than comparison operator.
                bool operator<(const Time& other) const { return nsecs_ < other.nsecs_; }
                
                //Greater than comparison operator.
                bool operator>(const Time& other) const { return nsecs_ > other.nsecs_; }

                //Less than or equal comparison operator.
                bool operator<=(const Time& other) const { return nsecs_ <= other.nsecs_; }

                //Greater than or equal comparison operator.
                bool operator>=(const Time& other) const { return nsecs_ >= other.nsecs_; }
            private:
                int64_t nsecs_;     //Stores time in nanoseconds for high precision.
        };

    } // namespace traj
} // namespace finalicp

// Specialization of std:hash function
namespace std {
    template <>
    struct hash<finalicp::traj::Time> {
        std::size_t operator()(const finalicp::traj::Time& k) const noexcept {
            // FNV-1a hash algorithm for better distribution
            std::size_t hash = 0xCBF29CE484222325ULL; // FNV-64 offset basis
            constexpr std::size_t prime = 0x100000001B3ULL; // FNV-64 prime

            // Mix the nanoseconds value
            hash ^= static_cast<std::size_t>(k.nanosecs());
            hash *= prime;

            // Additional mixing for better avalanche effect
            hash ^= hash >> 33;
            hash *= 0xFF51AFD7ED558CCDULL;
            hash ^= hash >> 33;
            hash *= 0xC4CEB9FE1A85EC53ULL;
            hash ^= hash >> 33;

            return hash;
        }
    };
} // namespace std