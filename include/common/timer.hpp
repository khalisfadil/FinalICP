#include <sys/time.h>
#include <iostream>

namespace finalicp {

    class Timer {
        public:
            
            //Default constructor
            Timer() {
                reset();
            }

            //Reset timer
            void reset() {
                beg_ = this->get_wall_time();
            }

            //Get seconds since last reset
            double seconds() const {
                return this->get_wall_time() - beg_;
            }

            //Get milliseconds since last reset
            double milliseconds() const {
                return 1000.0*(this->get_wall_time() - beg_);
            }

        private:

            //Get current wall time
            double get_wall_time() const {
                struct timeval time;
                if (gettimeofday(&time,NULL)){
                //  Handle error
                return 0;
                }
                return (double)time.tv_sec + (double)time.tv_usec * .000001;
            }

            //Wall time at reset
            double beg_;

    };

} // finalicp