#include <gtest/gtest.h>

#include <slam.hpp>

TEST(Slam, Time) {
  finalicp::traj::Time time;
  EXPECT_EQ(time.nanosecs(), 0);

  // example nanoseconds since epoch
  int64_t epoch64 = 1500000000123456789;
  //                     ^   sec  ^^ nano  ^

  // test epoch
  finalicp::traj::Time epoch = finalicp::traj::Time(epoch64);
  EXPECT_EQ(epoch.nanosecs(), 1500000000123456789);

  // test epoch
  int64_t epoch_secs64 = 1500000000e9;
  finalicp::traj::Time epoch_secs = finalicp::traj::Time(epoch_secs64);
  finalicp::traj::Time epoch_nsecs = epoch - epoch_secs;
  EXPECT_EQ(epoch_nsecs.nanosecs(), 123456789);

  // test double
  double epoch_nsecs_float = epoch_nsecs.seconds();
  EXPECT_NEAR(epoch_nsecs_float, 123456789e-9, 1e-6);

  // double back to Time
  finalicp::traj::Time nano_back_to_time(epoch_nsecs_float);
  EXPECT_EQ(nano_back_to_time.nanosecs(), 123456789);

  // two 32-bits to 64-bit (e.g. ros::Time)
  int32_t secs32 = 1500000000;
  int32_t nsecs32 = 123456789;
  EXPECT_EQ(epoch.nanosecs(), finalicp::traj::Time(secs32, nsecs32).nanosecs());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}